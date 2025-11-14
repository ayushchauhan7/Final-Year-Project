from flask import Blueprint, request, jsonify
from datetime import datetime
from bson import ObjectId
import re
from config.database import get_database
from utils.auth import hash_password, verify_password, generate_token

auth_bp = Blueprint('auth', __name__)

# Email validation regex
EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['username', 'email', 'password', 'fullName']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'{field} is required'}), 400
        
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        full_name = data['fullName'].strip()
        
        # Validate username
        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters long'}), 400
        
        if not re.match(r'^[a-zA-Z0-9_]+$', username):
            return jsonify({'error': 'Username can only contain letters, numbers, and underscores'}), 400
        
        # Validate email
        if not EMAIL_REGEX.match(email):
            return jsonify({'error': 'Invalid email format'}), 400
        
        # Validate password
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Get database
        db = get_database()
        
        # Check if username already exists
        if db.users.find_one({'username': username}):
            return jsonify({'error': 'Username already exists'}), 409
        
        # Check if email already exists
        if db.users.find_one({'email': email}):
            return jsonify({'error': 'Email already registered'}), 409
        
        # Hash password
        hashed_password = hash_password(password)
        
        # Create user document
        user_doc = {
            'username': username,
            'email': email,
            'password': hashed_password,
            'fullName': full_name,
            'role': 'user',
            'isActive': True,
            'createdAt': datetime.utcnow(),
            'lastLogin': None
        }
        
        # Insert user
        result = db.users.insert_one(user_doc)
        user_id = result.inserted_id
        
        # Log registration
        db.audit_logs.insert_one({
            'userId': user_id,
            'username': username,
            'action': 'register',
            'ipAddress': request.remote_addr,
            'userAgent': request.headers.get('User-Agent'),
            'timestamp': datetime.utcnow(),
            'details': {'email': email}
        })
        
        return jsonify({
            'message': 'Registration successful! Please login.',
            'username': username,
            'email': email
        }), 201
        
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed. Please try again.'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """Login user and return JWT token"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username and password are required'}), 400
        
        username = data['username'].strip()
        password = data['password']
        
        # Get database
        db = get_database()
        
        # Find user
        user = db.users.find_one({'username': username})
        
        if not user:
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Check if account is active
        if not user.get('isActive', True):
            return jsonify({'error': 'Account is deactivated. Please contact support.'}), 403
        
        # Verify password
        if not verify_password(password, user['password']):
            return jsonify({'error': 'Invalid username or password'}), 401
        
        # Update last login
        db.users.update_one(
            {'_id': user['_id']},
            {'$set': {'lastLogin': datetime.utcnow()}}
        )
        
        # Generate JWT token
        token = generate_token(user['_id'], user['username'], user['email'])
        
        # Log login
        db.audit_logs.insert_one({
            'userId': user['_id'],
            'username': user['username'],
            'action': 'login',
            'ipAddress': request.remote_addr,
            'userAgent': request.headers.get('User-Agent'),
            'timestamp': datetime.utcnow(),
            'details': {}
        })
        
        return jsonify({
            'message': 'Login successful!',
            'token': token,
            'user': {
                'id': str(user['_id']),
                'username': user['username'],
                'email': user['email'],
                'fullName': user['fullName'],
                'role': user.get('role', 'user')
            }
        }), 200
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': 'Login failed. Please try again.'}), 500

@auth_bp.route('/verify', methods=['GET'])
def verify_token():
    """Verify if token is valid"""
    from utils.auth import decode_token
    
    try:
        # Get token from header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'valid': False, 'error': 'No token provided'}), 401
        
        try:
            token = auth_header.split(" ")[1]
        except IndexError:
            return jsonify({'valid': False, 'error': 'Invalid token format'}), 401
        
        # Decode token
        payload = decode_token(token)
        
        if not payload:
            return jsonify({'valid': False, 'error': 'Invalid or expired token'}), 401
        
        return jsonify({
            'valid': True,
            'user': {
                'id': payload['user_id'],
                'username': payload['username'],
                'email': payload['email']
            }
        }), 200
        
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)}), 500

@auth_bp.route('/logout', methods=['POST'])
def logout():
    """Logout user (client-side token removal)"""
    from utils.auth import decode_token
    
    try:
        # Get token if available
        auth_header = request.headers.get('Authorization')
        if auth_header:
            try:
                token = auth_header.split(" ")[1]
                payload = decode_token(token)
                
                if payload:
                    # Log logout
                    db = get_database()
                    db.audit_logs.insert_one({
                        'userId': ObjectId(payload['user_id']),
                        'username': payload['username'],
                        'action': 'logout',
                        'ipAddress': request.remote_addr,
                        'userAgent': request.headers.get('User-Agent'),
                        'timestamp': datetime.utcnow(),
                        'details': {}
                    })
            except:
                pass
        
        return jsonify({'message': 'Logout successful'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500