import os
import time
import uuid
import gc
from flask import Flask, request, render_template, send_file, Response, jsonify
from TTS.api import TTS
from pydub import AudioSegment

# تنظیمات Flask
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# ساخت پوشه‌ها در صورت عدم وجود
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# صفحه اصلی
@app.route("/")
def index():
    return render_template("index.html")

# آپلود فایل صوتی
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    unique_id = str(uuid.uuid4())[:8]
    file_extension = file.filename.split(".")[-1]
    file_name = f"{unique_id}.{file_extension}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
    file.save(file_path)

    return jsonify({"status": "success", "message": "✅ File uploaded successfully!", "file_path": file_path, "unique_id": unique_id})

# پردازش متن و تولید گفتار **جمله به جمله**
@app.route("/process", methods=["POST"])
def process():
    text = request.form["text"]
    file_path = request.form["file_path"]
    unique_id = request.form["unique_id"]

    if not file_path or not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "Uploaded file not found"}), 400

    reference_wav = os.path.join(app.config["UPLOAD_FOLDER"], f"reference_{unique_id}.wav")
    output_wav = os.path.join(app.config["OUTPUT_FOLDER"], f"output_{unique_id}.wav")
    output_mp3 = os.path.join(app.config["OUTPUT_FOLDER"], f"output_{unique_id}.mp3")

    def generate():
        yield "🔄 Converting MP3 to WAV...\n"
        audio = AudioSegment.from_mp3(file_path)
        audio = audio.set_frame_rate(22050).set_channels(1)
        audio.export(reference_wav, format="wav", bitrate="128k")
        yield "✅ MP3 converted to WAV successfully!\n"
        time.sleep(1)

        yield "🔄 Loading TTS model...\n"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        yield "✅ Model loaded successfully!\n"
        time.sleep(1)

        yield "🔄 Generating speech sentence by sentence...\n"
        sentences = text.split(". ")  # تقسیم متن به جملات
        sentence_files = []

        for i, sentence in enumerate(sentences):
            output_wav_sentence = os.path.join(app.config["OUTPUT_FOLDER"], f"output_{unique_id}_{i}.wav")
            yield f"🔄 Processing sentence {i+1}/{len(sentences)}: {sentence}\n"

            # تولید گفتار برای هر جمله
            tts.tts_to_file(
                text=sentence,
                speaker_wav=reference_wav,
                language="en",
                file_path=output_wav_sentence
            )
            sentence_files.append(output_wav_sentence)

            yield f"✅ Sentence {i+1} processed!\n"
            time.sleep(0.5)  # جلوگیری از کرش شدن سرور در متن‌های طولانی

        yield "🔄 Combining all sentences into one WAV file...\n"
        combined_audio = AudioSegment.empty()

        for file in sentence_files:
            if os.path.exists(file):
                combined_audio += AudioSegment.from_wav(file)

        combined_audio.export(output_wav, format="wav")
        yield "✅ Final WAV file created!\n"
        time.sleep(1)

        yield "🔄 Converting WAV to MP3 (128kbps quality)...\n"
        audio = AudioSegment.from_wav(output_wav)
        audio.export(output_mp3, format="mp3", bitrate="128k")
        yield "✅ MP3 file ready for download!\n"

        # حذف فایل‌های موقت برای بهینه‌سازی حافظه
        for file in sentence_files:
            os.remove(file) if os.path.exists(file) else None
        os.remove(reference_wav) if os.path.exists(reference_wav) else None
        os.remove(output_wav) if os.path.exists(output_wav) else None
        gc.collect()

    return Response(generate(), mimetype="text/plain")

# دانلود فایل خروجی
@app.route("/download/<unique_id>")
def download(unique_id):
    output_mp3 = os.path.join(app.config["OUTPUT_FOLDER"], f"output_{unique_id}.mp3")
    if not os.path.exists(output_mp3):
        return "File not found", 404
    return send_file(output_mp3, as_attachment=True, download_name=f"output_{unique_id}.mp3")

# اجرای برنامه Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
