from flask import Flask, render_template, request, redirect
import os
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Configure upload folders
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_IMAGE = 'static/stylized_result.jpg'

# Clean and recreate upload folder
shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Routes ---

@app.route('/')
def index():
    return render_template('index01.html',
                           content_exists=os.path.exists(os.path.join(UPLOAD_FOLDER, 'content.jpg')),
                           style_exists=os.path.exists(os.path.join(UPLOAD_FOLDER, 'style.jpg')),
                           output_exists=os.path.exists(OUTPUT_IMAGE))


@app.route('/content_image', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return redirect('/')
    file = request.files['image']
    if file and file.filename:
        file.save(os.path.join(UPLOAD_FOLDER, 'content.jpg'))
    return redirect('/')


@app.route('/style_image', methods=['POST'])
def style_transfer():
    if 'style_image' not in request.files:
        return redirect('/')
    style_img = request.files['style_image']
    if style_img and style_img.filename:
        style_img.save(os.path.join(UPLOAD_FOLDER, 'style.jpg'))
    return redirect('/')


@app.route('/transfer', methods=['POST'])
def image_style_transfer():
    # Get epochs from form, default to 80 if missing or invalid
    try:
        num_iterations = int(request.form.get('epochs', 80))
        if num_iterations < 1:
            num_iterations = 1
        elif num_iterations > 400:
            num_iterations = 400
    except ValueError:
        num_iterations = 80
    
    content_weight = 1e3
    style_weight = 1e-2

    content_path = os.path.join(UPLOAD_FOLDER, 'content.jpg')
    style_path = os.path.join(UPLOAD_FOLDER, 'style.jpg')

    if not os.path.exists(content_path) or not os.path.exists(style_path):
        return redirect('/')

    def load_and_process_img(path):
        img = Image.open(path).convert('RGB').resize((224, 224))
        img = tf.keras.preprocessing.image.img_to_array(img)
        # print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")  # debug
        img = tf.keras.applications.vgg16.preprocess_input(img)
        return tf.convert_to_tensor(img[tf.newaxis, :], dtype=tf.float32)

    def deprocess_img(processed_img):
        x = processed_img.copy()[0]
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]
        return np.clip(x, 0, 255).astype('uint8')

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    def get_model():
        vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
        return tf.keras.Model([vgg.input], outputs)

    def gram_matrix(input_tensor):
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        return tf.matmul(a, a, transpose_a=True) / tf.cast(tf.shape(a)[0], tf.float32)

    def get_feature_representations(model, content_path, style_path):
        content_image = load_and_process_img(content_path)
        style_image = load_and_process_img(style_path)

        style_outputs = model(style_image)
        content_outputs = model(content_image)

        style_features = [gram_matrix(layer) for layer in style_outputs[:num_style_layers]]
        content_features = [layer for layer in content_outputs[num_style_layers:]]

        return style_features, content_features

    def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
        style_weight, content_weight = loss_weights
        model_outputs = model(init_image)

        style_output_features = model_outputs[:num_style_layers]
        content_output_features = model_outputs[num_style_layers:]

        style_score = tf.add_n([
            tf.reduce_mean(tf.square(gram_matrix(output) - target))
            for output, target in zip(style_output_features, gram_style_features)
        ]) * (style_weight / num_style_layers)

        content_score = tf.add_n([
            tf.reduce_mean(tf.square(output - target))
            for output, target in zip(content_output_features, content_features)
        ]) * (content_weight / num_content_layers)

        return style_score + content_score

    @tf.function()
    def train_step(image, model, optimizer, loss_weights, gram_style_features, content_features):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, loss_weights, image, gram_style_features, content_features)
        grad = tape.gradient(loss, image)
        optimizer.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image, -103.939, 255.0 - 103.939))
        return loss

    def run_style_transfer(content_path, style_path):
        model = get_model()
        style_features, content_features = get_feature_representations(model, content_path, style_path)

        init_image = load_and_process_img(content_path)
        init_image = tf.Variable(init_image, dtype=tf.float32)

        optimizer = tf.optimizers.Adam(learning_rate=5.0)
        best_img = None
        best_loss = float('inf')

        for i in range(num_iterations):
            loss = train_step(init_image, model, optimizer, (style_weight, content_weight),
                              style_features, content_features)
            if loss < best_loss:
                best_loss = loss
                best_img = init_image.numpy()
            if i % 50 == 0:
                print(f"Iteration {i}, Loss: {loss.numpy():.4f}")

        return deprocess_img(best_img)

    final_image = run_style_transfer(content_path, style_path)
    Image.fromarray(final_image).save(OUTPUT_IMAGE)

    return redirect('/')


if __name__ == '__main__':
    app.run(debug=True)
