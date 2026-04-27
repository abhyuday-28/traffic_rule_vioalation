# AI-Enabled Traffic Violation Detection System

This project is a desktop-based traffic rule violation detection system built with Python, OpenCV, Tkinter, and OCR models. It accepts image, video, or live camera input, detects violations, extracts vehicle plate details when possible, stores evidence, and supports reporting through email and Telegram.

## Current Features

- Image, video, and live camera input
- Triple riding detection
- No-helmet detection
- Red-light jumping detection
- Number plate extraction and OCR
- Evidence image saving
- Excel export of logged violations
- Email reporting
- Telegram reporting with photo attachment

## Project Structure

Main application files:

- `main.py`
- `traffic_system/gui.py`
- `traffic_system/pipeline.py`
- `traffic_system/models.py`
- `traffic_system/plate.py`
- `traffic_system/emailer.py`
- `traffic_system/settings.py`

Important model and asset paths:

- `_detector_source/object_detection_yolox_2022nov.onnx`
- `yolov3-obj.cfg`
- `yolov3-obj_2400.weights`
- `obj.names`
- `_ocr_source_fresh/wpod-net.json`
- `_ocr_source_fresh/wpod-net.h5`
- `_ocr_source_fresh/MobileNets_character_recognition.json`
- `_ocr_source_fresh/License_character_recognition_weight.h5`
- `_ocr_source_fresh/license_character_classes.npy`
- `_ocr_source_fresh/local_utils.py`

Sample input folders:

- `samples/images`
- `samples/videos`

Runtime output:

- `app_output`

## Python Version

Use Python `3.10`.

This project was built and tested with Python 3.10 and a local virtual environment.

## Setup

1. Clone or copy the project.
2. Install Git LFS if it is not already installed.
3. After cloning, run:

```powershell
git lfs install
git lfs pull
```

These two commands are required for this repository because the detector and OCR model files are stored using Git LFS.

4. Install Python `3.10`.
5. Open PowerShell in the project root.
6. Create a virtual environment:

```powershell
python -m venv venv
```

7. Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

8. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller python-pptx protobuf==3.20.3
```

## Running the App

Run the GUI with:

```powershell
.\venv\Scripts\python.exe main.py
```

If you want to use the helper batch files on another PC:

1. Double-click `setup_other_pc.bat`
2. Then double-click `run_app.bat`

## How the App Works

At a high level:

1. The GUI loads an image, video, or camera stream.
2. A YOLOX detector identifies traffic participants such as person, motorbike, car, truck, and traffic light.
3. A custom helmet detector checks helmet presence.
4. A rule engine determines violations such as:
   - triple riding
   - no helmet
   - red-light jumping
5. The vehicle crop is passed to the OCR pipeline for plate extraction and text recognition.
6. Evidence is saved in `app_output`.
7. Violations can be exported or reported.

## Telegram Reporting

The application supports sending the selected violation directly to Telegram using a bot.

### What it sends

When you choose a violation in the GUI and trigger Telegram reporting, the app sends:

- violation type
- timestamp
- detected plate text
- notes
- evidence image

### Where the Telegram code lives

Telegram configuration is defined in:

- `traffic_system/settings.py`

Telegram API sending logic is implemented in:

- `traffic_system/emailer.py`

The GUI button wiring is in:

- `traffic_system/gui.py`

### How Telegram technically works

The app does not use polling or webhooks. It simply makes an outbound HTTP request to the Telegram Bot API when the user clicks the Telegram report button.

If an evidence image exists, it uses the `sendPhoto` endpoint.
If no image exists, it falls back to `sendMessage`.

### How to connect your own Telegram bot

1. Open Telegram
2. Search for `@BotFather`
3. Start the chat
4. Send:

```text
/newbot
```

5. Give the bot a name
6. Give the bot a username ending with `bot`
7. BotFather will return a bot token

Example token format:

```text
123456789:AAExampleTokenHere
```

### How to get your chat ID

1. Open Telegram
2. Search for your new bot
3. Press `Start`
4. Send any message to the bot
5. Open this URL in a browser:

```text
https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
```

6. Look for:

```json
"chat": {
  "id": 123456789,
  "type": "private"
}
```

That `id` value is your chat ID.

### How to change Telegram settings in the project

Open:

- `traffic_system/settings.py`

Edit:

```python
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"
```

Then save the file and run the app again.

By default, the repository does not contain a real Telegram bot token or chat ID. You must add your own values before Telegram reporting will work.

### Group chat setup

If you want to send messages to a Telegram group:

1. Add the bot to the group
2. Send a message in the group
3. Call `getUpdates` again
4. Use the group chat ID returned by Telegram

Group IDs are usually negative numbers such as:

```text
-1001234567890
```

### Security note

Do not commit a real personal bot token to a public repository. If a token is exposed, revoke it with BotFather and generate a new one.

## Email Reporting

The application can also send a selected violation by email through SMTP.

This logic is implemented in:

- `traffic_system/emailer.py`

The GUI fields are in:

- `traffic_system/gui.py`

## Output Files

Generated outputs are stored in:

- `app_output`

Typical output includes:

- annotated evidence images
- cropped plate images
- exported Excel log

## Notes on Accuracy

This project is a working prototype, not a production-grade enforcement system.

Important limitations:

- helmet detection can still be inconsistent across viewpoints
- triple-riding detection is difficult in crowded scenes
- OCR depends heavily on image clarity and plate visibility
- long videos can be slow on CPU-only systems

## Troubleshooting

### App does not start

Make sure:

- Python `3.10` is installed
- the virtual environment was created successfully
- all dependencies are installed

### OCR says plate could not be read

That usually means:

- the crop was too small
- the plate was blurred
- the plate angle was poor
- OCR confidence was too low

### Telegram reporting fails

Check:

- bot token is correct
- chat ID is correct
- the bot has already received at least one message from that chat
- internet connection is available

## Suggested `.gitignore` Entries

If you are preparing this repo for GitHub, exclude generated and machine-local folders such as:

```gitignore
venv/
build/
dist/
app_output/
__pycache__/
*.pyc
```

## License

Use and distribute according to the license policy of your project or institution.
