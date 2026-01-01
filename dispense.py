import os
import time
import numpy as np
import cv2
import torch
import torchaudio

from insightface.app import FaceAnalysis
from utils.dose_lock import can_dispense
from utils.time_utils import get_current_slot
from configs.medicine_plans import MEDICINE_PLANS
from dispense_serial import dispense

EMB_FACES_DIR = "embeddings/faces"
EMB_VOICES_DIR = "embeddings/voices"

FACE_THRESHOLD = 0.50
VOICE_THRESHOLD = 0.60


def load_face_db(path=EMB_FACES_DIR):
    db = {}
    if not os.path.isdir(path):
        return db
    for f in os.listdir(path):
        if not f.lower().endswith(".npy"):
            continue
        name = os.path.splitext(f)[0]
        emb = np.load(os.path.join(path, f))
        db[name.lower()] = emb
    return db


def load_voice_db(path=EMB_VOICES_DIR):
    db = {}
    if not os.path.isdir(path):
        return db
    for f in os.listdir(path):
        if not (f.lower().endswith(".pt") or f.lower().endswith(".pth")):
            continue
        name = os.path.splitext(f)[0]
        emb = torch.load(os.path.join(path, f))
        db[name.lower()] = emb.squeeze()
    return db


def cosine_sim(a, b):
    a = a.flatten()
    b = b.flatten()
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def capture_face(timeout=5):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (index 0)")
    print("Look at the camera — capturing in 3 seconds...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture frame from camera")
    return frame


def get_face_embedding(img, model):
    faces = model.get(img)
    if not faces:
        return None
    return np.array(faces[0].embedding)


def record_voice(duration=3, out_path="/tmp/auth_voice.wav"):
    try:
        import sounddevice as sd
        import soundfile as sf
        sr = 16000
        print(f"Recording {duration}s of audio — please speak now...")

        # Try to auto-select a USB / camera microphone if available
        try:
            devs = sd.query_devices()
        except Exception:
            devs = []

        usb_dev_index = None
        for i, dev in enumerate(devs):
            try:
                if dev.get('max_input_channels', 0) > 0:
                    name = str(dev.get('name', '')).lower()
                    if any(k in name for k in ('usb', 'camera', 'webcam', 'logitech', 'uac')):
                        usb_dev_index = i
                        break
            except Exception:
                continue

        # Record using selected device if found, otherwise default device
        if usb_dev_index is not None:
            print(f"Using input device #{usb_dev_index}: {devs[usb_dev_index]['name']}")
            try:
                data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32', device=usb_dev_index)
                sd.wait()
            except Exception as e:
                print(f"Recording with selected device failed: {e}. Falling back to default device.")
                data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
                sd.wait()
        else:
            try:
                data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
                sd.wait()
            except Exception:
                raise

        # Ensure shape is (N, channels) for soundfile
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        sf.write(out_path, data, sr)
        print(f"Voice recording saved to {out_path}")
        return out_path
    except Exception as e:
        print(f"Microphone recording error: {e}")
        print("Attempting to list available audio devices:")
        try:
            import sounddevice as sd
            print(sd.query_devices())
        except Exception as dev_err:
            print(f"Could not list devices: {dev_err}")
        # fallback: ask user for a file path
        print("Could not record from microphone. Please provide a path to a WAV file:")
        p = input().strip()
        if not os.path.isfile(p):
            raise RuntimeError("Audio file not found")
        return p


def compute_mfcc_embedding(path, n_mfcc=40, target_sr=16000):
    # robust loader similar to train_voices
    try:
        audio, sr = torchaudio.load(path)
    except Exception:
        import soundfile as sf
        data, sr = sf.read(path, dtype='float32')
        data = np.asarray(data)
        if data.ndim == 1:
            audio = torch.from_numpy(data).unsqueeze(0)
        else:
            audio = torch.from_numpy(data.T)
    if audio.ndim > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    mfcc = torchaudio.transforms.MFCC(sample_rate=target_sr, n_mfcc=n_mfcc)(audio)
    emb = torch.mean(mfcc, dim=-1).squeeze().numpy()
    return emb


def match_face(db, emb):
    best = (None, -1.0)
    for name, d in db.items():
        s = cosine_sim(d, emb)
        if s > best[1]:
            best = (name, s)
    return best


def match_voice(db, emb):
    best = (None, -1.0)
    for name, d in db.items():
        d_np = d.cpu().numpy() if isinstance(d, torch.Tensor) else np.array(d)
        s = cosine_sim(d_np, emb)
        if s > best[1]:
            best = (name, s)
    return best


def run_once():
    face_db = load_face_db()
    voice_db = load_voice_db()

    if not face_db:
        print("No face embeddings found in embeddings/faces — run train_faces.py first")
        return

    # init face model
    app = FaceAnalysis(name="buffalo_s")
    try:
        app.prepare(ctx_id=0)
    except Exception:
        app.prepare(ctx_id=-1)

    try:
        frame = capture_face()
    except Exception as e:
        print("Face capture failed:", e)
        return

    face_emb = get_face_embedding(frame, app)
    if face_emb is None:
        print("No face detected — try again")
        return

    name, score = match_face(face_db, face_emb)
    print(f"Face match: {name} (score={score:.3f})")
    if name is None or score < FACE_THRESHOLD:
        print("Face not recognized or below threshold")
        return

    # Voice step
    if not voice_db:
        print("Warning: no voice embeddings found — voice check will use live recording if provided")

    audio_path = None
    try:
        audio_path = record_voice()
    except Exception as e:
        print("Voice recording failed:", e)
        return

    voice_emb = compute_mfcc_embedding(audio_path)
    vname, vscore = match_voice(voice_db, voice_emb)
    print(f"Voice best match: {vname} (score={vscore:.3f})")
    if vname is None or vscore < VOICE_THRESHOLD:
        print("Voice did not match any enrolled user or below threshold")
        return

    # Confirm same user
    if name != vname:
        print(f"Face and voice mismatch: face={name} voice={vname}")
        return

    user = name.lower()
    slot = get_current_slot()
    if not slot:
        print("Not a dispensing time")
        return

    if user not in MEDICINE_PLANS:
        print("User not configured in MEDICINE_PLANS")
        return

    plan = MEDICINE_PLANS[user].get(slot, [])
    if not plan:
        print(f"No medicines scheduled for {user} at {slot}")
        return

    if not can_dispense(user, slot):
        print("Dose already dispensed for this slot")
        return

    print(f"Authenticated {user}. Dispensing: {plan}")
    dispense(plan)


def main():
    print("Starting medication dispensing system...")
    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
        time.sleep(5)


if __name__ == '__main__':
    main()
