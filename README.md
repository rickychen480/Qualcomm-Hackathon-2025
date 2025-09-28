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

