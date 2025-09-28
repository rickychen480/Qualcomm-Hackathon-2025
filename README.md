# HomeEdge Security Assistant

**Contributors:**

* Ricky Chen - ricky.chen@princeton.edu
* Yousef Kassem - yk3904@princeton.edu
* Jessie Chen - chenjessie27@gmail.com

---

## **Project Overview**

HomeEdge is an **offline-first, edge-AI home security assistant** that runs entirely on a Windows laptop powered by Snapdragon® X Series processors. It continuously analyzes the built-in camera and microphone to detect people, unusual activity, and salient environmental sounds in real time — **without sending any data to the cloud**.

A configurable rolling pre/post-event buffer ensures that, when a threat is detected, the app commits both the **15-second lead-up** and the **aftermath** to **local, secure storage**.

---

## **Setup Instructions**

**1. Clone the repository**

```bash
git clone https://github.com/rickychen480/Qualcomm-Hackathon-2025.git
cd Qualcomm-Hackathon-2025
```

**2. Create a Python virtual environment**

*macOS/Linux:*

```bash
python3 -m venv venv
source venv/bin/activate
```

*Windows:*

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## **Run Instructions**

**4. Start the application**

```bash
streamlit run app.py
```

**5. Access the app**
Open your browser at `http://localhost:8501`. You will see the **HomeEdge Security Assistant** dashboard with three main tabs:

* **Control Dashboard** – Start/Stop detection, view recent alerts, and adjust detection sensitivity.
* **Archives** – View all past threat detection reports.
* **Settings** – Adjust buffer duration, storage settings, and recording quality.

**6. Using the app**

* Click **Start Detection** to begin monitoring.
* Alerts will appear in the **Recent Alerts** section and trigger popups.
* All detections are automatically stored in **Archives**.
* Use the **Settings** tab to configure system parameters.

**7. Stop the app**
Close the browser tab or press `Ctrl+C` in the terminal where Streamlit is running.

# License
MIT License

Copyright (c) 2025 Ricky Chen, Yousef Kassem, Jessie Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
