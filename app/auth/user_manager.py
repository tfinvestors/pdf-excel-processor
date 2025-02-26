import os
import sqlite3
import hashlib
import secrets
import logging
import json
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("auth.log")
    ]
)
logger = logging.getLogger("auth")


class UserManager:
    def __init__(self, db_path='data/users.db'):
        """Initialize the UserManager with the path to the user database."""
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        self.initialize_db()

    def initialize_db(self):
        """Create the users table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT,
                is_admin INTEGER DEFAULT 0,
                created_at TEXT,
                last_login TEXT,
                is_active INTEGER DEFAULT 1
            )
        ''')

        # Create sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                session_token TEXT UNIQUE NOT NULL,
                created_at TEXT,
                expires_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        # Create user_activity table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_activity (
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                activity_type TEXT NOT NULL,
                details TEXT,
                timestamp TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')

        conn.commit()
        conn.close()

        # Create an admin user if none exists
        self._create_default_admin()

    def _create_default_admin(self):
        """Create a default admin user if no users exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if any users exist
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]

        if count == 0:
            # Create a default admin user
            admin_username = "admin"
            admin_email = "admin@example.com"
            admin_password = "admin123"  # This should be changed after first login

            # Create the admin user
            self.register_user(
                username=admin_username,
                email=admin_email,
                password=admin_password,
                full_name="Administrator",
                is_admin=1
            )

            logger.info("Created default admin user. Please change the password after first login.")

        conn.close()

    def _hash_password(self, password, salt=None):
        """
        Hash a password with a salt using SHA-256.

        Args:
            password (str): The password to hash
            salt (str, optional): The salt to use. If None, a new salt is generated.

        Returns:
            tuple: (password_hash, salt)
        """
        if salt is None:
            # Generate a random salt
            salt = secrets.token_hex(16)

        # Hash the password with the salt
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()

        return password_hash, salt

    def register_user(self, username, email, password, full_name=None, is_admin=0):
        """
        Register a new user.

        Args:
            username (str): The username
            email (str): The email address
            password (str): The password
            full_name (str, optional): The user's full name
            is_admin (int, optional): Whether the user is an admin (1) or not (0)

        Returns:
            bool: True if registration was successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if username or email already exists
            cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
            if cursor.fetchone():
                logger.warning(f"Username or email already exists: {username}, {email}")
                return False, "Username or email already exists"

            # Hash the password
            password_hash, salt = self._hash_password(password)

            # Insert the new user
            created_at = datetime.datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, salt, full_name, is_admin, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (username, email, password_hash, salt, full_name, is_admin, created_at))

            # Get the user ID
            user_id = cursor.lastrowid

            # Log the registration
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (user_id, "REGISTER", "User registered", created_at))

            conn.commit()
            logger.info(f"User registered successfully: {username}")
            return True, user_id

        except Exception as e:
            logger.error(f"Error registering user: {str(e)}")
            return False, str(e)

        finally:
            conn.close()

    def authenticate_user(self, username, password):
        """
        Authenticate a user.

        Args:
            username (str): The username or email
            password (str): The password

        Returns:
            tuple: (success, user_data or error_message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Find the user by username or email
            cursor.execute('''
                SELECT id, username, email, password_hash, salt, full_name, is_admin, is_active 
                FROM users 
                WHERE (username = ? OR email = ?)
            ''', (username, username))

            user_data = cursor.fetchone()

            if not user_data:
                return False, "Invalid username or password"

            user_id, db_username, email, password_hash, salt, full_name, is_admin, is_active = user_data

            if not is_active:
                return False, "Account is inactive"

            # Verify the password
            computed_hash, _ = self._hash_password(password, salt)

            if computed_hash != password_hash:
                return False, "Invalid username or password"

            # Update last login time
            last_login = datetime.datetime.now().isoformat()
            cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (last_login, user_id))

            # Log the login
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (user_id, "LOGIN", "User logged in", last_login))

            # Create a session
            session_token = secrets.token_hex(32)
            expires_at = (datetime.datetime.now() + datetime.timedelta(days=1)).isoformat()

            cursor.execute('''
                INSERT INTO sessions (user_id, session_token, created_at, expires_at)
                VALUES (?, ?, ?, ?)
            ''', (user_id, session_token, last_login, expires_at))

            conn.commit()

            # Return user data as a dictionary
            user_dict = {
                "id": user_id,
                "username": db_username,
                "email": email,
                "full_name": full_name,
                "is_admin": bool(is_admin),
                "session_token": session_token
            }

            return True, user_dict

        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}")
            return False, str(e)

        finally:
            conn.close()

    def verify_session(self, session_token):
        """
        Verify a session token.

        Args:
            session_token (str): The session token

        Returns:
            tuple: (is_valid, user_data or error_message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Find the session
            cursor.execute('''
                SELECT s.user_id, s.expires_at, u.username, u.email, u.full_name, u.is_admin, u.is_active
                FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.session_token = ?
            ''', (session_token,))

            session_data = cursor.fetchone()

            if not session_data:
                return False, "Invalid session"

            user_id, expires_at, username, email, full_name, is_admin, is_active = session_data

            # Check if session is expired
            if datetime.datetime.fromisoformat(expires_at) < datetime.datetime.now():
                # Delete expired session
                cursor.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
                conn.commit()
                return False, "Session expired"

            # Check if user is active
            if not is_active:
                return False, "Account is inactive"

            # Return user data
            user_data = {
                "id": user_id,
                "username": username,
                "email": email,
                "full_name": full_name,
                "is_admin": bool(is_admin)
            }

            return True, user_data

        except Exception as e:
            logger.error(f"Error verifying session: {str(e)}")
            return False, str(e)

        finally:
            conn.close()

    def logout_user(self, session_token):
        """
        Log out a user by invalidating their session token.

        Args:
            session_token (str): The session token

        Returns:
            bool: True if logout was successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Find the user ID for the session
            cursor.execute("SELECT user_id FROM sessions WHERE session_token = ?", (session_token,))
            result = cursor.fetchone()

            if result:
                user_id = result[0]

                # Delete the session
                cursor.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))

                # Log the logout
                timestamp = datetime.datetime.now().isoformat()
                cursor.execute('''
                    INSERT INTO user_activity (user_id, activity_type, details, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, "LOGOUT", "User logged out", timestamp))

                conn.commit()
                return True

            return False

        except Exception as e:
            logger.error(f"Error logging out user: {str(e)}")
            return False

        finally:
            conn.close()

    def change_password(self, user_id, current_password, new_password):
        """
        Change a user's password.

        Args:
            user_id (int): The user ID
            current_password (str): The current password
            new_password (str): The new password

        Returns:
            bool: True if password change was successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get the user's current password hash and salt
            cursor.execute("SELECT password_hash, salt FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()

            if not result:
                return False, "User not found"

            current_hash, salt = result

            # Verify the current password
            computed_hash, _ = self._hash_password(current_password, salt)

            if computed_hash != current_hash:
                return False, "Current password is incorrect"

            # Hash the new password
            new_hash, new_salt = self._hash_password(new_password)

            # Update the password
            cursor.execute('''
                UPDATE users 
                SET password_hash = ?, salt = ? 
                WHERE id = ?
            ''', (new_hash, new_salt, user_id))

            # Log the password change
            timestamp = datetime.datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (user_id, "PASSWORD_CHANGE", "Password changed", timestamp))

            conn.commit()
            return True, "Password changed successfully"

        except Exception as e:
            logger.error(f"Error changing password: {str(e)}")
            return False, str(e)

        finally:
            conn.close()

    def get_user_list(self, admin_user_id):
        """
        Get a list of all users (admin only).

        Args:
            admin_user_id (int): The ID of the admin user making the request

        Returns:
            tuple: (success, list of users or error message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if the requesting user is an admin
            cursor.execute("SELECT is_admin FROM users WHERE id = ?", (admin_user_id,))
            result = cursor.fetchone()

            if not result or not result[0]:
                return False, "Access denied. Admin privileges required."

            # Get all users
            cursor.execute('''
                SELECT id, username, email, full_name, is_admin, created_at, last_login, is_active
                FROM users
            ''')

            users = []
            for row in cursor.fetchall():
                user_id, username, email, full_name, is_admin, created_at, last_login, is_active = row
                users.append({
                    "id": user_id,
                    "username": username,
                    "email": email,
                    "full_name": full_name,
                    "is_admin": bool(is_admin),
                    "created_at": created_at,
                    "last_login": last_login,
                    "is_active": bool(is_active)
                })

            return True, users

        except Exception as e:
            logger.error(f"Error getting user list: {str(e)}")
            return False, str(e)

        finally:
            conn.close()

    def update_user_status(self, admin_user_id, target_user_id, is_active):
        """
        Activate or deactivate a user (admin only).

        Args:
            admin_user_id (int): The ID of the admin user making the request
            target_user_id (int): The ID of the user to update
            is_active (bool): Whether the user should be active

        Returns:
            tuple: (success, message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if the requesting user is an admin
            cursor.execute("SELECT is_admin FROM users WHERE id = ?", (admin_user_id,))
            result = cursor.fetchone()

            if not result or not result[0]:
                return False, "Access denied. Admin privileges required."

            # Update the user status
            cursor.execute('''
                UPDATE users 
                SET is_active = ? 
                WHERE id = ?
            ''', (1 if is_active else 0, target_user_id))

            # Log the action
            status = "activated" if is_active else "deactivated"
            timestamp = datetime.datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (admin_user_id, "USER_STATUS_CHANGE", f"User {target_user_id} {status}", timestamp))

            conn.commit()
            return True, f"User {status} successfully"

        except Exception as e:
            logger.error(f"Error updating user status: {str(e)}")
            return False, str(e)

        finally:
            conn.close()

    def log_activity(self, user_id, activity_type, details):
        """
        Log user activity.

        Args:
            user_id (int): The user ID
            activity_type (str): The type of activity
            details (str): Details about the activity

        Returns:
            bool: True if logging was successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = datetime.datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO user_activity (user_id, activity_type, details, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (user_id, activity_type, details, timestamp))

            conn.commit()
            return True

        except Exception as e:
            logger.error(f"Error logging activity: {str(e)}")
            return False

        finally:
            conn.close()