import helpers
import os
import time
from datetime import datetime
import streamlit as st


class ReportProcessor:
    def __init__(self, recorder, session_state):
        self.recorder = recorder
        self.st_state = session_state

    def process(self, detection_data):
        # Generate report
        video_path = helpers.find_latest_video(os.path.join(os.getcwd(), self.recorder.output_dir))
        recording_data = self.get_recording_data(video_path)
        report = {
            "id": len(self.st_state.archived_reports) + 1,
            "threat_type": detection_data.get("threat_type", "unknown"),
            "timestamp": datetime.now(),
            "threat_type": detection_data["threat_type"],
            "confidence": detection_data["confidence"],
            "bounding_boxes": detection_data.get("bounding_boxes", []),
            "performance_metrics": detection_data.get("performance_metrics", {}),
            "recording_data": recording_data,
            "summary": f"{detection_data['threat_type'].title()} detection with {detection_data['confidence']:.1%} confidence",
            "video_path": video_path,
            "description": f"Detected threat: {detection_data.get('threat_type', 'unknown')}. Immediate attention required.",
            "severity": detection_data.get("severity", "High"),
            "actions_taken": ["Alert sent to homeowner", "Video recorded"],
        }
        
        print(f"[Reporter] Report generated: {report}")
        return report

    def get_recording_data(self, video_path):
        """Get recording data after saving replay"""

        # Get file statistics (size and creation time)
        file_stats = os.stat(video_path)
        file_size_mb = file_stats.st_size / (1024 * 1024)
        creation_time = datetime.fromtimestamp(file_stats.st_ctime).strftime("%m/%d/%y - %H:%M:%S")

        # Populate the dictionary with real data from the file
        recording_data = {
            "timestamp": creation_time,
            "duration_minutes": f"{self.st_state.storage_settings['buffer_duration']} minutes",
            "path": video_path,
            "size_mb": round(file_size_mb, 2),
        }
        return recording_data