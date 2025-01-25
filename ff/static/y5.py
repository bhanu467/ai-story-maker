import uuid
file_ext = file.filename.split('.')[-1]
unique_filename = f"{uuid.uuid4()}.{file_ext}"
file.save(os.path.join('static/uploads', unique_filename))
