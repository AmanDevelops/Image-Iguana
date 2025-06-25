from flask import Blueprint, render_template, request, flash, redirect, url_for, send_file, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import base64
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import zipfile
import tempfile
import shutil

image_processing = Blueprint('image_processing', __name__)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'webp', 'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_format(img, filename, format_conversion):
    base = filename.rsplit('.', 1)[0]
    match format_conversion:
        case "cwebp":
            out = os.path.join("static", f"{base}.webp")  # FIX 1: Use os.path.join for cross-platform
            cv2.imwrite(out, img)
            return out
        case "cpng":
            out = os.path.join("static", f"{base}.png")
            cv2.imwrite(out, img)
            return out
        case "cjpg":
            out = os.path.join("static", f"{base}.jpg")
            cv2.imwrite(out, img)
            return out
        case "cjpeg":
            out = os.path.join("static", f"{base}.jpeg")
            cv2.imwrite(out, img)
            return out
    return None

def apply_processing(img, filename, image_processing):
    base = filename.rsplit('.', 1)[0]
    match image_processing:
        case "cgray":
            img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out = os.path.join("static", f"{base}_gray.png")
            cv2.imwrite(out, img_processed)
            return out, img_processed
        case "histeq":
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_processed = cv2.equalizeHist(img_gray)
            out = os.path.join("static", f"{base}_histeq.png")
            cv2.imwrite(out, img_processed)
            return out, img_processed
        case "blur":
            img_processed = cv2.GaussianBlur(img, (5, 5), 0)
            out = os.path.join("static", f"{base}_blurred.png")
            cv2.imwrite(out, img_processed)
            return out, img_processed
        case "canny":
            img_processed = cv2.Canny(img, 100, 200)
            out = os.path.join("static", f"{base}_edges.png")
            cv2.imwrite(out, img_processed)
            return out, img_processed
        case "rotate":
            img_processed = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            out = os.path.join("static", f"{base}_rotated.png")
            cv2.imwrite(out, img_processed)
            return out, img_processed
        case "sharpen":
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img_processed = cv2.filter2D(img, -1, kernel)
            out = os.path.join("static", f"{base}_sharpened.png")
            cv2.imwrite(out, img_processed)
            return out, img_processed
    return None, img

def generate_output_filename(base, new_format):
    return os.path.join("static", f"{base}_processed.{new_format}")  # FIX 2: Use os.path.join

def processImage(filename, format_conversion=None, image_processing=None):
    img_path = os.path.join("uploads", filename)  # FIX 3: Use os.path.join for uploads
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return None

    # Format conversion only
    if format_conversion and not image_processing:
        return convert_format(img, filename, format_conversion)

    # Image processing only or both
    base = filename.rsplit('.', 1)[0]
    ext = filename.rsplit('.', 1)[1].lower()
    img_processed = img
    out_path = None

    if image_processing:
        out_path, img_processed = apply_processing(img, filename, image_processing)
        if not out_path:
            out_path = os.path.join("static", f"{base}_processed.{ext}")
    else:
        out_path = os.path.join("static", f"{base}_processed.{ext}")

    # If format conversion is also requested after processing
    if format_conversion:
        temp_path = os.path.join("static", f"{base}_temp.{ext}")
        cv2.imwrite(temp_path, img_processed)
        img2 = cv2.imread(temp_path)
        out_path = convert_format(img2, f"{base}_processed.{ext}", format_conversion)
        if os.path.exists(temp_path):  # FIX 4: Check before removing
            os.remove(temp_path)
        return out_path

    cv2.imwrite(out_path, img_processed)
    return out_path

@image_processing.route('/edit', methods=['GET', 'POST'])
@login_required
def edit():
    if request.method == 'POST':
        if 'annotated_image' in request.form and request.form['annotated_image']:
            data_url = request.form['annotated_image']
            original_filename = request.form['original_filename']
            edited_filename = request.form['edited_filename']
            header, encoded = data_url.split(",", 1)
            data = base64.b64decode(encoded)
            img = Image.open(BytesIO(data))
            annotated_path = os.path.join("static", "uploads", "annotated_" + os.path.basename(edited_filename))
            img.save(annotated_path)
            return render_template(
                "preview.html",
                original_filename=original_filename,
                edited_filename=os.path.relpath(annotated_path, "static").replace("\\", "/")
            )
        format_conversion = request.form.get("format_conversion")
        image_processing_opt = request.form.get("image_processing")
        if 'file' not in request.files:
            flash('No file part in request')
            return redirect(url_for('image_processing.edit'))
        files = request.files.getlist('file')
        if len(files) == 0 or files[0].filename == '':
            flash('No files selected for upload')
            return redirect(url_for('image_processing.edit'))
        processed_files = []
        error_files = []
        for file in files:
            try:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    processed_file = processImage(filename, format_conversion, image_processing_opt)
                    if processed_file:
                        processed_files.append(processed_file)
                    else:
                        error_files.append(f"{filename} (processing failed)")
                else:
                    error_files.append(f"{file.filename} (invalid type)")
            except Exception as e:
                error_files.append(f"{file.filename} (error: {str(e)})")  # FIX 5: Add missing closing parenthesis
        if error_files:
            flash(f"Errors with {len(error_files)} file(s): {', '.join(error_files[:3])}{'...' if len(error_files) > 3 else ''}")
        if not processed_files:
            flash('No files were processed successfully')
            return redirect(url_for('image_processing.edit'))
        if len(processed_files) == 1:
            download_filename = os.path.basename(processed_files[0])
            abs_path = os.path.abspath(processed_files[0])
            if not os.path.exists(abs_path):
                flash(f'Processed file not found: {download_filename}')
                return render_template("error.html")
            return send_file(
                abs_path,
                as_attachment=True,
                download_name=download_filename,
                mimetype='image/png'
            )
        else:
            try:
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, 'processed_images.zip')
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file_path in processed_files:
                        zipf.write(file_path, os.path.basename(file_path))
                def cleanup():
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        print(f"Cleanup error: {e}")
                response = send_file(
                    zip_path,
                    as_attachment=True,
                    download_name='processed_images.zip',
                    mimetype='application/zip'
                )
                response.call_on_close(cleanup)
                return response
            except Exception as e:
                flash(f'Error creating zip file: {str(e)}')
                return redirect(url_for('image_processing.edit'))
    return render_template("index.html")

@image_processing.route('/download/<path:filename>')
@login_required
def download(filename):
    file_path = os.path.join("static", filename)
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        flash(f'File not found: {filename}')
        return render_template("error.html")
    return send_file(abs_path, as_attachment=True)
