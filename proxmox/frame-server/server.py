#!/usr/bin/env python3
"""
Frame Server for Letta MAS Video Quality Evaluation
Provides REST API for extracting frames and listing videos from ComfyUI output.

Run: python server.py
Endpoint: http://0.0.0.0:8189
"""

from flask import Flask, jsonify, send_file, request
from flask_cors import CORS
import subprocess
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
COMFYUI_OUTPUT = os.environ.get('COMFYUI_OUTPUT', '/home/straughter/ComfyUI/output/video')
TEMP_DIR = '/tmp/frames'

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)


@app.route('/')
def index():
    """Health check and API info."""
    return jsonify({
        'service': 'Frame Server',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            '/list_videos': 'GET - List all videos',
            '/extract_frame/<filename>/<frame_num>': 'GET - Extract specific frame',
            '/video/<filename>': 'GET - Serve video file',
            '/video_info/<filename>': 'GET - Get video metadata'
        }
    })


@app.route('/list_videos')
def list_videos():
    """List all video files in ComfyUI output directory."""
    try:
        if not os.path.exists(COMFYUI_OUTPUT):
            return jsonify({'error': f'Output directory not found: {COMFYUI_OUTPUT}', 'videos': []})

        videos = []
        for f in os.listdir(COMFYUI_OUTPUT):
            if f.endswith('.mp4'):
                filepath = os.path.join(COMFYUI_OUTPUT, f)
                stat = os.stat(filepath)
                videos.append({
                    'filename': f,
                    'size_bytes': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'path': filepath
                })

        # Sort by modification time, newest first
        videos.sort(key=lambda x: x['modified'], reverse=True)

        return jsonify({
            'count': len(videos),
            'directory': COMFYUI_OUTPUT,
            'videos': videos
        })
    except Exception as e:
        return jsonify({'error': str(e), 'videos': []}), 500


@app.route('/extract_frame/<filename>/<int:frame_num>')
def extract_frame(filename, frame_num):
    """Extract a specific frame from a video."""
    try:
        video_path = os.path.join(COMFYUI_OUTPUT, filename)

        if not os.path.exists(video_path):
            return jsonify({'error': f'Video not found: {filename}'}), 404

        output_filename = f'{filename.replace(".mp4", "")}_{frame_num}.png'
        output_path = os.path.join(TEMP_DIR, output_filename)

        # Extract frame using ffmpeg
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', f'select=eq(n\\,{frame_num})',
            '-vframes', '1',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0 or not os.path.exists(output_path):
            return jsonify({
                'error': 'Failed to extract frame',
                'stderr': result.stderr
            }), 500

        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/video/<filename>')
def serve_video(filename):
    """Serve a video file directly."""
    try:
        video_path = os.path.join(COMFYUI_OUTPUT, filename)

        if not os.path.exists(video_path):
            return jsonify({'error': f'Video not found: {filename}'}), 404

        return send_file(video_path, mimetype='video/mp4')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/video_info/<filename>')
def video_info(filename):
    """Get metadata about a video using ffprobe."""
    try:
        video_path = os.path.join(COMFYUI_OUTPUT, filename)

        if not os.path.exists(video_path):
            return jsonify({'error': f'Video not found: {filename}'}), 404

        # Get video info using ffprobe
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return jsonify({'error': 'Failed to get video info'}), 500

        info = json.loads(result.stdout)

        # Extract useful info
        video_stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), {})

        return jsonify({
            'filename': filename,
            'duration': float(info.get('format', {}).get('duration', 0)),
            'size_bytes': int(info.get('format', {}).get('size', 0)),
            'width': video_stream.get('width'),
            'height': video_stream.get('height'),
            'fps': eval(video_stream.get('r_frame_rate', '0/1')) if video_stream.get('r_frame_rate') else None,
            'codec': video_stream.get('codec_name'),
            'total_frames': int(video_stream.get('nb_frames', 0)) if video_stream.get('nb_frames') else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print(f"Starting Frame Server on port 8189")
    print(f"Video directory: {COMFYUI_OUTPUT}")
    app.run(host='0.0.0.0', port=8189, debug=False)
