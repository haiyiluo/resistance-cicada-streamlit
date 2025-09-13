"""
Fix the import issues in server.py
"""
import os
import shutil

print("FIXING SERVER.PY IMPORTS")
print("="*60)

# Find server.py
server_locations = [
    "cascade-main/resistance_cascade/server.py",
    "resistance_cascade/server.py"
]

server_file = None
for loc in server_locations:
    if os.path.exists(loc):
        server_file = loc
        break

if not server_file:
    print("❌ Cannot find server.py!")
    exit(1)

print(f"Found server.py at: {server_file}")

# Read the current content
with open(server_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Create backup
backup_file = server_file + ".backup"
shutil.copy(server_file, backup_file)
print(f"Created backup: {backup_file}")

# Fix the imports
fixed_content = content.replace(
    """from .model import ResistanceCascade
from .agent import Citizen, Security""",
    """# Fix imports - try multiple methods
try:
    # If running as part of package
    from .model import ResistanceCascade
    from .agent import Citizen, Security
except ImportError:
    # If running directly
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from model import ResistanceCascade
        from agent import Citizen, Security
    except ImportError:
        from resistance_cascade.model import ResistanceCascade
        from resistance_cascade.agent import Citizen, Security""")

# Add the __main__ block if not present
if 'if __name__ == "__main__"' not in fixed_content:
    fixed_content += """

# Add this to make it runnable
if __name__ == "__main__":
    server.port = 8521
    print(f"Starting ResistanceCascade visualization server...")
    print(f"Open your browser to: http://localhost:{server.port}/")
    server.launch()
"""

# Write the fixed content
with open(server_file, 'w', encoding='utf-8') as f:
    f.write(fixed_content)

print("✓ Fixed imports in server.py")
print("\nNow you can run the server with:")
print(f"  python {server_file}")
print("\nOr restore the original with:")
print(f"  copy {backup_file} {server_file}")