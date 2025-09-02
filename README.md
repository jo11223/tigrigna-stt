# **🎤 Whisper Tigrigna STT: Fine-Tuned Speech-to-Text Model**

Unlock the power of **voice in Tigrigna**! This project fine-tunes **OpenAI Whisper** using **LoRA (Low-Rank Adaptation)** to transcribe **Tigrigna speech into text** with high accuracy. Perfect for transcription, accessibility tools, and voice-driven applications.

---

## **✨ Key Features**

* 🚀 Fine-tuned **Whisper-small** for Tigrigna
* 💡 Uses **LoRA** for efficient GPU memory usage
* 🗂️ Handles custom Tigrigna datasets with TSV + audio format
* 📊 Tracks **training loss** and **Word Error Rate (WER)**
* 🏁 Exports a **final LoRA-adapted model** for instant inference
* 🎯 Provides **sample transcription examples** for quick testing

---

## **🛠️ Setup & Dependencies**

Install required packages:

```bash
!pip install git+https://github.com/openai/whisper.git
!pip install torchaudio librosa datasets jiwer evaluate
!apt-get install -y ffmpeg
```

Additional libraries:

* **Transformers (Hugging Face)** – tokenizer, feature extractor, Whisper model
* **PEFT (LoRA)** – efficient fine-tuning
* **Torch + CUDA** – GPU training
* **Google Drive (Colab)** – store datasets and checkpoints
* **Matplotlib, tqdm, pandas, numpy** – data handling & visualization

---

## **📂 Dataset Format**

Your dataset should contain:

* **TSV file** (`data.tsv`) with `path` & `sentence` columns
* **Audio folder** (`recordings/`) containing `.wav` or `.flac` files

Example TSV:

```text
path    sentence
audio1.wav    ሰላም ከመይ ኣሎ?
audio2.wav    መልእክት ላክልኝ።
```

Update paths in the script:

```python
dataset_dir = 'your-dataset-path'
shared_drive_dir = 'your-shared-drive-path'
```

---

## **🏋️ Training Process**

1. Mount **Google Drive** to store checkpoints & final model
2. Prepare dataset using Hugging Face `Dataset` API
3. Tokenize audio-text pairs and extract features
4. Apply **LoRA configuration** for memory efficiency
5. Train for **8 epochs** using AdamW optimizer
6. Auto-save checkpoints after each epoch
7. Evaluate **WER** on test set
8. Visualize **loss curves** and **token length distributions**

---

## **💾 Checkpointing & Resume Training**

* Saves model & optimizer states along with training loss
* Resume training seamlessly from the last saved checkpoint

---

## **🏁 Export Final Model**

```python
model.save_pretrained(os.path.join(shared_drive_dir, "lora_model_tigrigna_final"))
```

Your LoRA-adapted Whisper model is now ready for deployment or further fine-tuning.

---

## **🔍 Inference Example**

Run transcription on test samples:

```text
▶ Sample 0:
True : ሰላም ከመይ ኣሎ?
Pred.: ሰላም ከመይ ኣሎ
```

Test with your own audio files by updating the inference section.

---

## **📊 Visualization**

* **Training Loss Curve** – track model improvement
* **Token Length Histogram** – see sentence length distribution

---

## **🎯 Project Goal**

This fine-tuned Whisper model is designed to **bring Tigrigna speech to text** with high accuracy, opening possibilities for:

* Voice transcription
* Accessibility tools
* Voice-controlled apps
* Language research & documentation

