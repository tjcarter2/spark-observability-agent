import browser_cookie3
import subprocess
import os
import glob

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# PASTE YOUR PROFILE PATH HERE (From chrome://version)
# specific_profile_path = r"C:\Users\Name\AppData\Local\Google\Chrome\User Data\Profile 2"  # Windows Example
# specific_profile_path = "/Users/Name/Library/Application Support/Google/Chrome/Profile 2"  # Mac Example

# Set this to None if you want to try the default, otherwise paste the path above inside the quotes
PROFILE_PATH = r"/Users/Name/Library/Application Support/Google/Chrome/Profile 2" 

# Target Settings
DOMAIN_KEYWORD = "" 
COOKIE_NAME = "DATAPLANE_DOMAIN_DBAUTH"
SECRET_SCOPE = "shscreds"
SECRET_KEY = "cookies"

# Set this to your databricks CLI profile name (ex: logfood)
DB_CLI_PROFILE = "prod"

# ==============================================================================

def get_cookie_file_path(profile_path):
    """
    Constructs the full path to the 'Cookies' file based on the profile folder.
    """
    # Windows/Linux usually just name the file 'Cookies'
    # Mac sometimes uses 'Cookies' or 'Google/Chrome/Default/Cookies'
    
    # Try exact path first (if user pointed directly to the file)
    if os.path.isfile(profile_path):
        return profile_path

    # Try standard filenames inside the profile folder
    possible_names = ["Cookies", "Network/Cookies"] 
    
    for name in possible_names:
        full_path = os.path.join(profile_path, name)
        if os.path.exists(full_path):
            return full_path
            
    print(f"⚠️ Warning: Could not find a file named 'Cookies' inside {profile_path}")
    print("Trying to use the folder anyway...")
    return None

def update_databricks_secret(cookie_value):
    print(f"\n🚀 Updating secret '{SECRET_KEY}' in scope '{SECRET_SCOPE}'...")
    try:
        command = [
            "databricks", "secrets", "put-secret", 
            SECRET_SCOPE, SECRET_KEY, 
            "--string-value", cookie_value, "-p", DB_CLI_PROFILE
        ]
        subprocess.run(command, check=True)
        print("✅ Secret updated successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to update secret. Check CLI configuration.")
    except FileNotFoundError:
        print("❌ 'databricks' CLI command not found.")

def main():
    cookie_file = None
    if PROFILE_PATH and "PASTE_YOUR_PATH" not in PROFILE_PATH:
        print(f"📂 Targeting specific profile: {PROFILE_PATH}")
        cookie_file = get_cookie_file_path(PROFILE_PATH)
    else:
        print("📂 Using Default Chrome Profile...")

    try:
        # Load cookies from specific file if provided
        if cookie_file:
            cj = browser_cookie3.chrome(cookie_file=cookie_file)
        else:
            cj = browser_cookie3.chrome()
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        return

    found_cookie = None
    target_domain = None

    print(f"🔍 Scanning cookies for '{COOKIE_NAME}'...")
    
    count = 0
    for cookie in cj:
        count += 1
        if cookie.name == COOKIE_NAME:
            if DOMAIN_KEYWORD in cookie.domain:
                found_cookie = cookie.value
                target_domain = cookie.domain
                break
    
    print(f"📊 Scanned {count} cookies.")

    if found_cookie:
        print(f"\n✅ FOUND COOKIE!")
        print(f"   Domain: {target_domain}")
        print(f"Cookie: {found_cookie}")
        update_databricks_secret(found_cookie)
    else:
        print(f"\n❌ Could not find cookie in this profile.")
        print("   Are you sure you logged into the Spark History Server on *this* profile?")

if __name__ == "__main__":
    main()
