#!/usr/bin/env python3
"""
Suspect Lock API Endpoints for Foresight SAR

Provides REST API endpoints for suspect/victim locking functionality.

Endpoints:
- POST /suspect - Upload reference image(s), returns target_id
- GET /suspect/{id}/matches - Returns live matches with locations
- PUT /suspect/{id}/lock - Lock/unlock suspect
- DELETE /suspect/{id} - Remove suspect from tracking
- GET /suspect/{id} - Get suspect details
- GET /suspects - List all suspects
- POST /suspect/{id}/verify/{match_id} - Verify/reject a match
- GET /audit/{target_id} - Get audit log for target
"""

import os
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, current_app
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError

from suspect_lock import SuspectLockManager, SuspectTarget, SuspectMatch, AuditLogEntry


class SuspectAPI:
    """Flask API for suspect lock functionality"""
    
    def __init__(self, app: Flask, storage_dir: str = "data/suspects", model_path: Optional[str] = None):
        self.app = app
        self.manager = SuspectLockManager(storage_dir, model_path)
        self.upload_dir = Path(storage_dir) / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure upload settings
        self.app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
        self.allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        
        self._register_routes()
    
    def _allowed_file(self, filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def _save_uploaded_file(self, file) -> str:
        """Save uploaded file and return path"""
        if not file or file.filename == '':
            raise BadRequest("No file provided")
        
        if not self._allowed_file(file.filename):
            raise BadRequest(f"File type not allowed. Allowed types: {', '.join(self.allowed_extensions)}")
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        file_path = self.upload_dir / filename
        
        file.save(str(file_path))
        return str(file_path)
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.route('/api/suspect', methods=['POST'])
        def create_suspect():
            """Create new suspect target with reference images"""
            try:
                # Get form data
                name = request.form.get('name')
                description = request.form.get('description')
                priority = request.form.get('priority', 'medium')
                tags = request.form.get('tags', '').split(',') if request.form.get('tags') else []
                user_id = request.form.get('user_id')
                
                # Validate priority
                if priority not in ['low', 'medium', 'high', 'critical']:
                    return jsonify({'error': 'Invalid priority. Must be: low, medium, high, critical'}), 400
                
                # Get uploaded files
                if 'images' not in request.files:
                    return jsonify({'error': 'No images provided'}), 400
                
                files = request.files.getlist('images')
                if not files or all(f.filename == '' for f in files):
                    return jsonify({'error': 'No images provided'}), 400
                
                # Save uploaded files
                image_paths = []
                for file in files:
                    if file and file.filename != '':
                        try:
                            file_path = self._save_uploaded_file(file)
                            image_paths.append(file_path)
                        except Exception as e:
                            return jsonify({'error': f'Error saving file {file.filename}: {str(e)}'}), 400
                
                if not image_paths:
                    return jsonify({'error': 'No valid images provided'}), 400
                
                # Create suspect target
                target_id = self.manager.create_suspect_target(
                    reference_images=image_paths,
                    name=name,
                    description=description,
                    priority=priority,
                    tags=[tag.strip() for tag in tags if tag.strip()],
                    user_id=user_id
                )
                
                return jsonify({
                    'target_id': target_id,
                    'message': 'Suspect target created successfully',
                    'reference_images_count': len(image_paths)
                }), 201
            
            except Exception as e:
                current_app.logger.error(f"Error creating suspect: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/suspect/<target_id>', methods=['GET'])
        def get_suspect(target_id: str):
            """Get suspect target details"""
            try:
                if target_id not in self.manager.targets:
                    return jsonify({'error': 'Suspect target not found'}), 404
                
                target = self.manager.targets[target_id]
                
                # Convert to dict for JSON serialization
                target_data = {
                    'target_id': target.target_id,
                    'name': target.name,
                    'description': target.description,
                    'created_timestamp': target.created_timestamp.isoformat(),
                    'is_locked': target.is_locked,
                    'lock_timestamp': target.lock_timestamp.isoformat() if target.lock_timestamp else None,
                    'locked_by': target.locked_by,
                    'priority': target.priority,
                    'tags': target.tags,
                    'reference_images_count': len(target.reference_images)
                }
                
                return jsonify(target_data), 200
            
            except Exception as e:
                current_app.logger.error(f"Error getting suspect {target_id}: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/suspects', methods=['GET'])
        def list_suspects():
            """List all suspect targets"""
            try:
                # Get query parameters
                locked_only = request.args.get('locked_only', 'false').lower() == 'true'
                priority = request.args.get('priority')
                tag = request.args.get('tag')
                limit = int(request.args.get('limit', 100))
                
                targets = []
                for target in self.manager.targets.values():
                    # Apply filters
                    if locked_only and not target.is_locked:
                        continue
                    if priority and target.priority != priority:
                        continue
                    if tag and tag not in target.tags:
                        continue
                    
                    target_data = {
                        'target_id': target.target_id,
                        'name': target.name,
                        'description': target.description,
                        'created_timestamp': target.created_timestamp.isoformat(),
                        'is_locked': target.is_locked,
                        'priority': target.priority,
                        'tags': target.tags,
                        'reference_images_count': len(target.reference_images)
                    }
                    targets.append(target_data)
                
                # Sort by creation time (newest first)
                targets.sort(key=lambda x: x['created_timestamp'], reverse=True)
                
                return jsonify({
                    'targets': targets[:limit],
                    'total_count': len(targets)
                }), 200
            
            except Exception as e:
                current_app.logger.error(f"Error listing suspects: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/suspect/<target_id>/matches', methods=['GET'])
        def get_suspect_matches(target_id: str):
            """Get live matches for a suspect target"""
            try:
                if target_id not in self.manager.targets:
                    return jsonify({'error': 'Suspect target not found'}), 404
                
                # Get query parameters
                limit = int(request.args.get('limit', 100))
                verified_only = request.args.get('verified_only', 'false').lower() == 'true'
                min_confidence = float(request.args.get('min_confidence', 0.0))
                
                matches = self.manager.get_suspect_matches(target_id, limit * 2)  # Get more to filter
                
                # Apply filters
                filtered_matches = []
                for match in matches:
                    if verified_only and not match.verified:
                        continue
                    if match.confidence < min_confidence:
                        continue
                    
                    match_data = {
                        'match_id': match.match_id,
                        'target_id': match.target_id,
                        'confidence': match.confidence,
                        'timestamp': match.timestamp.isoformat(),
                        'location': match.location,
                        'bounding_box': match.bounding_box,
                        'frame_id': match.frame_id,
                        'camera_id': match.camera_id,
                        'verified': match.verified,
                        'verified_by': match.verified_by,
                        'verification_timestamp': match.verification_timestamp.isoformat() if match.verification_timestamp else None
                    }
                    filtered_matches.append(match_data)
                
                return jsonify({
                    'matches': filtered_matches[:limit],
                    'total_count': len(filtered_matches)
                }), 200
            
            except Exception as e:
                current_app.logger.error(f"Error getting matches for {target_id}: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/suspect/<target_id>/lock', methods=['PUT'])
        def lock_suspect(target_id: str):
            """Lock or unlock a suspect target"""
            try:
                if target_id not in self.manager.targets:
                    return jsonify({'error': 'Suspect target not found'}), 404
                
                data = request.get_json()
                if not data or 'locked' not in data:
                    return jsonify({'error': 'Missing locked field in request body'}), 400
                
                locked = bool(data['locked'])
                user_id = data.get('user_id')
                
                success = self.manager.lock_suspect(target_id, locked, user_id)
                
                if success:
                    return jsonify({
                        'message': f"Suspect {'locked' if locked else 'unlocked'} successfully",
                        'target_id': target_id,
                        'locked': locked
                    }), 200
                else:
                    return jsonify({'error': 'Failed to update lock status'}), 500
            
            except Exception as e:
                current_app.logger.error(f"Error locking suspect {target_id}: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/suspect/<target_id>', methods=['DELETE'])
        def delete_suspect(target_id: str):
            """Delete a suspect target"""
            try:
                if target_id not in self.manager.targets:
                    return jsonify({'error': 'Suspect target not found'}), 404
                
                user_id = request.args.get('user_id')
                
                success = self.manager.delete_suspect_target(target_id, user_id)
                
                if success:
                    return jsonify({
                        'message': 'Suspect target deleted successfully',
                        'target_id': target_id
                    }), 200
                else:
                    return jsonify({'error': 'Failed to delete suspect target'}), 500
            
            except Exception as e:
                current_app.logger.error(f"Error deleting suspect {target_id}: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/suspect/<target_id>/verify/<match_id>', methods=['POST'])
        def verify_match(target_id: str, match_id: str):
            """Verify or reject a suspect match"""
            try:
                if target_id not in self.manager.targets:
                    return jsonify({'error': 'Suspect target not found'}), 404
                
                data = request.get_json()
                if not data or 'verified' not in data:
                    return jsonify({'error': 'Missing verified field in request body'}), 400
                
                verified = bool(data['verified'])
                user_id = data.get('user_id')
                
                success = self.manager.verify_match(target_id, match_id, verified, user_id)
                
                if success:
                    return jsonify({
                        'message': f"Match {'verified' if verified else 'rejected'} successfully",
                        'target_id': target_id,
                        'match_id': match_id,
                        'verified': verified
                    }), 200
                else:
                    return jsonify({'error': 'Match not found'}), 404
            
            except Exception as e:
                current_app.logger.error(f"Error verifying match {match_id} for {target_id}: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/suspect/search', methods=['POST'])
        def search_suspects():
            """Search for suspects matching uploaded image"""
            try:
                # Get uploaded file
                if 'image' not in request.files:
                    return jsonify({'error': 'No image provided'}), 400
                
                file = request.files['image']
                if file.filename == '':
                    return jsonify({'error': 'No image provided'}), 400
                
                # Get parameters
                threshold = float(request.form.get('threshold', 0.7))
                
                # Save temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    file.save(tmp_file.name)
                    
                    # Load image
                    image = cv2.imread(tmp_file.name)
                    if image is None:
                        os.unlink(tmp_file.name)
                        return jsonify({'error': 'Invalid image file'}), 400
                    
                    # Search for matches
                    matches = self.manager.search_suspects(image, threshold)
                    
                    # Clean up temporary file
                    os.unlink(tmp_file.name)
                
                # Format results
                results = []
                for target_id, similarity in matches:
                    target = self.manager.targets[target_id]
                    results.append({
                        'target_id': target_id,
                        'name': target.name,
                        'similarity': similarity,
                        'priority': target.priority,
                        'is_locked': target.is_locked
                    })
                
                return jsonify({
                    'matches': results,
                    'threshold': threshold,
                    'total_count': len(results)
                }), 200
            
            except Exception as e:
                current_app.logger.error(f"Error searching suspects: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/audit/<target_id>', methods=['GET'])
        def get_audit_log(target_id: str):
            """Get audit log for a suspect target"""
            try:
                if target_id not in self.manager.targets:
                    return jsonify({'error': 'Suspect target not found'}), 404
                
                limit = int(request.args.get('limit', 100))
                
                entries = self.manager.get_audit_log(target_id, limit)
                
                audit_data = []
                for entry in entries:
                    audit_data.append({
                        'entry_id': entry.entry_id,
                        'timestamp': entry.timestamp.isoformat(),
                        'action': entry.action,
                        'target_id': entry.target_id,
                        'user_id': entry.user_id,
                        'details': entry.details,
                        'ip_address': entry.ip_address,
                        'user_agent': entry.user_agent
                    })
                
                return jsonify({
                    'audit_entries': audit_data,
                    'total_count': len(audit_data)
                }), 200
            
            except Exception as e:
                current_app.logger.error(f"Error getting audit log for {target_id}: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/api/audit', methods=['GET'])
        def get_all_audit_logs():
            """Get all audit log entries"""
            try:
                limit = int(request.args.get('limit', 100))
                action = request.args.get('action')
                user_id = request.args.get('user_id')
                
                entries = self.manager.get_audit_log(None, limit * 2)  # Get more to filter
                
                # Apply filters
                filtered_entries = []
                for entry in entries:
                    if action and entry.action != action:
                        continue
                    if user_id and entry.user_id != user_id:
                        continue
                    
                    filtered_entries.append({
                        'entry_id': entry.entry_id,
                        'timestamp': entry.timestamp.isoformat(),
                        'action': entry.action,
                        'target_id': entry.target_id,
                        'user_id': entry.user_id,
                        'details': entry.details,
                        'ip_address': entry.ip_address,
                        'user_agent': entry.user_agent
                    })
                
                return jsonify({
                    'audit_entries': filtered_entries[:limit],
                    'total_count': len(filtered_entries)
                }), 200
            
            except Exception as e:
                current_app.logger.error(f"Error getting audit logs: {str(e)}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.errorhandler(413)
        def too_large(e):
            return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413
        
        @self.app.errorhandler(400)
        def bad_request(e):
            return jsonify({'error': 'Bad request'}), 400
        
        @self.app.errorhandler(404)
        def not_found(e):
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(e):
            return jsonify({'error': 'Internal server error'}), 500


# Example usage
if __name__ == "__main__":
    from flask import Flask
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
    
    # Initialize suspect API
    suspect_api = SuspectAPI(app)
    
    # Add CORS support for development
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
    
    # Health check endpoint
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'suspect-lock-api'
        }), 200
    
    # API documentation endpoint
    @app.route('/api/docs', methods=['GET'])
    def api_docs():
        return jsonify({
            'endpoints': {
                'POST /api/suspect': 'Create new suspect target with reference images',
                'GET /api/suspect/{id}': 'Get suspect target details',
                'GET /api/suspects': 'List all suspect targets',
                'GET /api/suspect/{id}/matches': 'Get live matches for suspect',
                'PUT /api/suspect/{id}/lock': 'Lock/unlock suspect target',
                'DELETE /api/suspect/{id}': 'Delete suspect target',
                'POST /api/suspect/{id}/verify/{match_id}': 'Verify/reject a match',
                'POST /api/suspect/search': 'Search suspects by image',
                'GET /api/audit/{target_id}': 'Get audit log for target',
                'GET /api/audit': 'Get all audit logs',
                'GET /api/health': 'Health check',
                'GET /api/docs': 'API documentation'
            }
        }), 200
    
    print("Starting Suspect Lock API server...")
    print("API Documentation: http://localhost:5000/api/docs")
    print("Health Check: http://localhost:5000/api/health")
    
    app.run(host='0.0.0.0', port=5000, debug=True)