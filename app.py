from flask import Flask, render_template, request, send_file
from rembg import remove
from PIL import Image
from io import BytesIO
from pytubefix import YouTube
from pathlib import Path
import base64
import re
from demucs import pretrained
from demucs.apply import apply_model
import torchaudio
import torch
import logging
from pydub import AudioSegment
import shutil

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('homepage.html')

def convert_img_format(image_file, frmat):
    with Image.open(image_file) as img:
        output_img = io.BytesIO()
        img.save(output_img, format=frmat.upper())
        output_img.seek(0)
        return output_img

@app.route('/convert', methods=['GET', 'POST'])
def convert():
    if request.method == 'POST':
        if 'image_file' not in request.files or not request.files['image_file']:
            return "No image uploaded!", 400

        image_file = request.files['image_file']
        output_format = request.form.get('format')

        if not output_format:
            return "No format selected!", 400

        try:
            converted_image = convert_img_format(image_file, output_format)
            return send_file(
                converted_image,
                mimetype=f'image/{output_format}',
                as_attachment=True,
                download_name=f'converted_image.{output_format}'
            )
        except Exception as e:
            return f"An error occurred: {e}", 500

    return render_template('convert.html')

@app.route('/rmbg', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            input_image = Image.open(file.stream)
        # Check if a base64 image is provided
        elif 'base64_image' in request.form and request.form['base64_image']:
            base64_image_data = request.form['base64_image']
            # Strip the prefix (e.g., "data:image/png;base64,")
            base64_image_data = re.sub(r"^data:image/\w+;base64,", "", base64_image_data)
            image_data = base64.b64decode(base64_image_data)
            input_image = Image.open(BytesIO(image_data))
        else:
            return 'No valid image provided', 400

        # Process the image with rembg
        output_image = remove(input_image, post_process_mask=True)
        img_io = BytesIO()
        output_image.save(img_io, 'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png', as_attachment=True, download_name='_rmbg.png')

    return render_template('index.html')

@app.route("/yt2mp4", methods=["GET", "POST"])
def downloadVideo():
    mesage = ''
    errorType = 0
    if request.method == 'POST' and 'video_url' in request.form:
        youtubeUrl = request.form["video_url"]
        action = request.form.get("action")  # Get action from button
        if youtubeUrl:
            validateVideoUrl = (
                r'(https?://)?(www\.)?'
                '(youtube|youtu|youtube-nocookie)\.(com|be)/'
                '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
            validVideoUrl = re.match(validateVideoUrl, youtubeUrl)
            if validVideoUrl:
                try:
                    url = YouTube(youtubeUrl)
                    if action == "video":
                        video = url.streams.get_highest_resolution()
                        video_buffer = BytesIO()
                        video.stream_to_buffer(video_buffer)
                        video_buffer.seek(0)
                        return send_file(
                            video_buffer,
                            mimetype="video/mp4",
                            as_attachment=True,
                            download_name=f"{url.title}.mp4"
                        )
                    elif action == "audio":
                        # Select audio-only stream
                        audio = url.streams.filter(only_audio=True, file_extension="mp3").first()
                        if not audio:
                            audio = url.streams.filter(only_audio=True).first()  # Fallback
                        
                        audio_buffer = BytesIO()
                        audio.stream_to_buffer(audio_buffer)
                        audio_buffer.seek(0)
                        return send_file(
                            audio_buffer,
                            mimetype="audio/mp3", 
                            as_attachment=True,
                            download_name=f"{url.title}.mp3"
                        )
                except Exception as e:
                    mesage = f"Error processing request: {str(e)}"
                    errorType = 0
            else:
                mesage = 'Enter a valid YouTube video URL!'
                errorType = 0
        else:
            mesage = 'Enter a YouTube video URL.'
            errorType = 0
    return render_template('youtube.html', mesage=mesage, errorType=errorType)

def convert_audio_format(input_path, output_path, output_format):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format=output_format)

@app.route('/audio_converter', methods=['GET', 'POST'])
def audio_converter():
    if request.method == 'POST':
        if 'audio_file' not in request.files or not request.files['audio_file']:
            return "No audio file uploaded!", 400

        audio_file = request.files['audio_file']
        output_format = request.form.get('format')

        if not output_format:
            return "No format selected!", 400

        input_audio_path = Path("uploaded_audio") / audio_file.filename
        input_audio_path.parent.mkdir(exist_ok=True)
        audio_file.save(input_audio_path)

        output_audio_path = input_audio_path.with_suffix(f".{output_format}")
        try:
            convert_audio_format(input_audio_path, output_audio_path, output_format)
            return send_file(
                output_audio_path,
                mimetype=f'audio/{output_format}',
                as_attachment=True,
                download_name=f'converted_audio.{output_format}'
            )
        except Exception as e:
            return f"An error occurred: {e}", 500

    return render_template('audio_converter.html')

@app.route('/demucs', methods=['GET', 'POST'])
def demucs_separate():
    if request.method == 'POST':
        if 'audio_file' not in request.files or not request.files['audio_file']:
            return "No audio file uploaded!", 400

        audio_file = request.files['audio_file']
        input_audio_path = Path("uploaded_audio") / audio_file.filename
        input_audio_path.parent.mkdir(exist_ok=True)
        audio_file.save(input_audio_path)

        # Convert to WAV if not already in WAV format
        if input_audio_path.suffix.lower() != '.wav':
            wav_path = input_audio_path.with_suffix('.wav')
            try:
                audio = AudioSegment.from_file(input_audio_path)
                audio.export(wav_path, format='wav')
                input_audio_path = wav_path
            except Exception as e:
                return f"Error converting audio to WAV: {e}", 500

        output_dir = Path("demucs_output") / input_audio_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            torchaudio.set_audio_backend("sox_io")
            model = pretrained.get_model('htdemucs')
            model.cpu()

            wav, sample_rate = torchaudio.load(str(input_audio_path))
            sources = apply_model(model, wav, split=True, overlap=0.25, shifts=1)

            stems = ['vocals', 'drums', 'bass', 'other']
            for source, stem in zip(sources, stems):
                output_path = output_dir / f"{stem}.wav"
                torchaudio.save(str(output_path), source.cpu(), sample_rate)
        except Exception as e:
            return f"An error occurred while processing the audio: {e}", 500

        zip_path = output_dir.with_suffix(".zip")
        try:
            shutil.make_archive(str(output_dir), 'zip', str(output_dir))
            return send_file(
                str(zip_path),
                mimetype='application/zip',
                as_attachment=True,
                download_name=f'{input_audio_path.stem}_stems.zip'
            )
        except Exception as e:
            return f"An error occurred while creating the zip file: {e}", 500

    return render_template('demucs.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5100)
