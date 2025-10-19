from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
import hmac
import hashlib
import os
import requests
import json
import base64
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai

load_dotenv()

app = FastAPI(
    title="GitHub AI Agent",
    description="AI-powered GitHub automation agent",
    version="1.0.0"
)

WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"🔮 Gemini AI configured: {bool(GEMINI_API_KEY)}")
else:
    print("⚠️  Gemini API key not found - AI analysis will be disabled")

print(f"🔧 Environment loaded: GITHUB_TOKEN length = {len(GITHUB_TOKEN) if GITHUB_TOKEN else 0}")

class GitHubAPI:
    def __init__(self, token: str):
        self.token = token
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.base_url = "https://api.github.com"
    
    def get_file_content(self, repo: str, file_path: str, ref: str = "main"):
        """Get file content from GitHub at specific commit"""
        url = f"{self.base_url}/repos/{repo}/contents/{file_path}?ref={ref}"
        try:
            print(f"   🔍 Making GitHub API request to: {url}")
            response = requests.get(url, headers=self.headers)
            print(f"   📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 404:
                print(f"   ⚠️  File not found: {file_path} at ref {ref}")
                return None
            elif response.status_code == 403:
                print(f"   🔐 Rate limit or permission issue: {response.text}")
                return None
            else:
                print(f"   ❌ GitHub API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"   ❌ GitHub API Exception: {e}")
            return None

# Initialize GitHub API
github_api = GitHubAPI(GITHUB_TOKEN)

# ===== CORE ENDPOINTS =====
@app.get("/")
async def root():
    return {"message": "GitHub AI Agent is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "github-ai-agent"}

@app.get("/debug-env")
async def debug_env():
    """Check environment variables"""
    print("🔍 Debug env endpoint called")
    return {
        "webhook_secret_set": bool(WEBHOOK_SECRET),
        "webhook_secret_preview": WEBHOOK_SECRET[:10] + "..." if WEBHOOK_SECRET else "None",
        "github_token_set": bool(GITHUB_TOKEN),
        "github_token_preview": GITHUB_TOKEN[:10] + "..." if GITHUB_TOKEN else "None",
        "github_token_length": len(GITHUB_TOKEN) if GITHUB_TOKEN else 0,
        "message": "Environment variables loaded successfully!"
    }

@app.get("/debug-token")
async def debug_token():
    """Debug endpoint to check GitHub token"""
    print("🔐 Debug token endpoint called")
    if not GITHUB_TOKEN:
        return {"error": "No GitHub token configured"}
    
    # Test the token by making a simple API call
    url = "https://api.github.com/user"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers)
        print(f"🔐 GitHub API response status: {response.status_code}")
        if response.status_code == 200:
            user_data = response.json()
            return {
                "token_valid": True,
                "user": user_data.get("login"),
                "rate_limit_remaining": response.headers.get("X-RateLimit-Remaining"),
                "message": "Token is valid!"
            }
        else:
            return {
                "token_valid": False,
                "status_code": response.status_code,
                "error": response.text
            }
    except Exception as e:
        return {"error": str(e)}

@app.get("/routes")
async def list_routes():
    """List all available routes"""
    routes = []
    for route in app.routes:
        route_info = {
            "path": getattr(route, "path", None),
            "methods": getattr(route, "methods", None),
            "name": getattr(route, "name", None)
        }
        routes.append(route_info)
    return {"available_routes": routes}

# ===== WEBHOOK FUNCTIONALITY =====
def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature"""
    if not WEBHOOK_SECRET or not signature:
        return True
    
    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode(), 
        payload, 
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected_signature}", signature)

@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    x_hub_signature_256: str = Header(None),
    x_github_event: str = Header(None)
):
    """
    Receive GitHub webhooks
    """
    try:
        # Get raw payload
        payload = await request.body()
        payload_str = payload.decode('utf-8')
        
        # Verify webhook signature
        if WEBHOOK_SECRET and x_hub_signature_256:
            if not verify_webhook_signature(payload, x_hub_signature_256):
                raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        # Parse JSON payload
        json_payload = json.loads(payload_str)
        
        print(f"📨 Received GitHub event: {x_github_event}")
        
        # Handle different GitHub events
        if x_github_event == "push":
            return await handle_push_event(json_payload)
        elif x_github_event == "pull_request":
            return await handle_pull_request_event(json_payload)
        else:
            return JSONResponse(
                status_code=200,
                content={"message": f"Event {x_github_event} received"}
            )
            
    except Exception as e:
        print(f"❌ Webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_push_event(payload: dict):
    """Handle GitHub push events with ACTUAL file content analysis"""
    repo_name = payload["repository"]["full_name"]
    commits = payload.get("commits", [])
    before_commit = payload.get("before")
    after_commit = payload.get("after")
    
    print(f"🚀 Push event for {repo_name}")
    print(f"📊 Commits: {len(commits)}")
    
    # Extract ALL changed files from all commits
    all_changed_files = set()
    for commit in commits:
        all_changed_files.update(commit.get("added", []))
        all_changed_files.update(commit.get("modified", []))
        all_changed_files.update(commit.get("removed", []))
    
    changed_files = list(all_changed_files)
    print(f"📄 Changed files: {changed_files}")
    
    # Get ACTUAL file changes and content
    file_changes_analysis = await analyze_file_changes(repo_name, commits, changed_files)
    
    # Get AI analysis if Gemini is configured
    ai_analysis_result = {}
    if commits and GEMINI_API_KEY:
        commit_message = commits[-1].get("message", "No commit message")
        ai_analysis_result = await analyze_changes_with_gemini(
            file_changes_analysis, 
            repo_name, 
            commit_message
        )
    else:
        ai_analysis_result = {"ai_analysis": "Gemini not configured or no commits"}
    
    return {
        "status": "processed",
        "event": "push",
        "repo": repo_name,
        "commit_count": len(commits),
        "file_changes": file_changes_analysis,
        "ai_analysis": ai_analysis_result
    }

async def handle_pull_request_event(payload: dict):
    """Handle GitHub pull request events"""
    pr_action = payload.get("action")
    pr_number = payload["pull_request"]["number"]
    repo_name = payload["repository"]["full_name"]
    
    print(f"🔀 PR #{pr_number} {pr_action} for {repo_name}")
    
    return {
        "status": "processed", 
        "event": "pull_request",
        "action": pr_action,
        "pr_number": pr_number,
        "repo": repo_name
    }

async def analyze_file_changes(repo_name: str, commits: list, changed_files: list) -> list:
    """Get ACTUAL file content and changes"""
    print(f"🔍 Getting actual content for {len(changed_files)} changed files...")
    
    file_analysis = []
    
    for file_path in changed_files:
        print(f"\n   📁 ===== Analyzing: {file_path} =====")
        
        file_info = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "status": "modified",
            "before_content": None,
            "after_content": None,
            "before_size": 0,
            "after_size": 0,
            "before_lines": 0,
            "after_lines": 0
        }
        
        if commits:
            current_commit_sha = commits[-1]["id"]  # The new commit
            parent_commit_sha = commits[0].get("parents", [{}])[0].get("sha") if commits[0].get("parents") else None
            
            print(f"   🔄 Current commit: {current_commit_sha[:8]}")
            print(f"   🔄 Parent commit: {parent_commit_sha[:8] if parent_commit_sha else 'None (likely initial commit)'}")
            
            before_commit_sha = commits[0].get("id")  # This might be wrong approach
            
            if len(commits) > 0:
                first_commit_sha = commits[0]["id"]
                parent_commit_sha = await get_parent_commit(repo_name, first_commit_sha)
                
                if parent_commit_sha:
                    print(f"   🔍 Getting BEFORE content from parent: {parent_commit_sha[:8]}")
                    before_content = await get_file_content_at_commit(repo_name, file_path, parent_commit_sha)
                    if before_content:
                        file_info["before_content"] = before_content["content"]
                        file_info["before_size"] = before_content["size"]
                        file_info["before_lines"] = before_content["lines"]
                        print(f"   ✅ BEFORE content: {file_info['before_size']} chars, {file_info['before_lines']} lines")
                    else:
                        print(f"   ⚠️  No BEFORE content found - file might be newly added")
                else:
                    print(f"   ⚠️  No parent commit - this might be the initial commit")
            
            # Get AFTER content (current state)
            print(f"   🔍 Getting AFTER content from current: {current_commit_sha[:8]}")
            after_content = await get_file_content_at_commit(repo_name, file_path, current_commit_sha)
            if after_content:
                file_info["after_content"] = after_content["content"]
                file_info["after_size"] = after_content["size"]
                file_info["after_lines"] = after_content["lines"]
                print(f"   ✅ AFTER content: {file_info['after_size']} chars, {file_info['after_lines']} lines")
            else:
                print(f"   ⚠️  No AFTER content found - file might be deleted")
            
            # Determine file status from commit data
            for commit in commits:
                if file_path in commit.get("added", []):
                    file_info["status"] = "added"
                    print(f"   📝 Status: ADDED (from commit data)")
                    break
                elif file_path in commit.get("removed", []):
                    file_info["status"] = "removed" 
                    print(f"   📝 Status: REMOVED (from commit data)")
                    break
                elif file_path in commit.get("modified", []):
                    file_info["status"] = "modified"
                    print(f"   📝 Status: MODIFIED (from commit data)")
                    break
            
            # If we couldn't determine from commit data, use content presence
            if file_info["status"] == "modified":
                if not file_info["before_content"] and file_info["after_content"]:
                    file_info["status"] = "added"
                    print(f"   📝 Status: ADDED (inferred from content)")
                elif file_info["before_content"] and not file_info["after_content"]:
                    file_info["status"] = "removed"
                    print(f"   📝 Status: REMOVED (inferred from content)")
        
        # Show content comparison
        print(f"   📊 FINAL CONTENT SUMMARY:")
        print(f"     - Status: {file_info['status']}")
        print(f"     - Before: {file_info['before_size']} chars, {file_info['before_lines']} lines")
        print(f"     - After: {file_info['after_size']} chars, {file_info['after_lines']} lines")
        
        if file_info["before_content"] and file_info["after_content"]:
            content_diff = len(file_info['after_content']) - len(file_info['before_content'])
            print(f"     - Size change: {content_diff:+d} chars")
            
            # Show preview of changes
            if content_diff != 0:
                print(f"     - Before preview: {file_info['before_content'][:100]}...")
                print(f"     - After preview: {file_info['after_content'][:100]}...")
        
        file_analysis.append(file_info)
    
    print(f"\n✅ File content analysis complete for {len(file_analysis)} files")
    return file_analysis

async def get_file_content_at_commit(repo_name: str, file_path: str, commit_sha: str) -> dict:
    """Get file content at specific commit"""
    if not GITHUB_TOKEN:
        print(f"   ❌ GitHub token not available")
        return None
    
    if not commit_sha:
        print(f"   ❌ No commit SHA provided")
        return None
    
    try:
        print(f"   🔍 Fetching content for '{file_path}' at commit {commit_sha[:8]}...")
        file_data = github_api.get_file_content(repo_name, file_path, commit_sha)
        
        if file_data and "content" in file_data:
            # Decode base64 content
            content = base64.b64decode(file_data["content"]).decode('utf-8')
            print(f"   ✅ Successfully fetched {len(content)} characters")
            
            return {
                "content": content,
                "size": len(content),
                "lines": content.count('\n') + 1,
                "encoding": file_data.get("encoding", "base64"),
                "sha": file_data.get("sha", "")
            }
        else:
            if file_data is None:
                print(f"   ⚠️  File not found or no content (might be deleted)")
            elif "message" in file_data:
                print(f"   ❌ GitHub API error: {file_data['message']}")
            else:
                print(f"   ❌ Unknown error fetching content")
            
    except Exception as e:
        print(f"   ❌ Exception fetching content: {str(e)}")
    
    return None

async def get_parent_commit(repo_name: str, commit_sha: str) -> str:
    """Get the parent commit SHA for a given commit"""
    if not GITHUB_TOKEN:
        print(f"   ❌ No GitHub token available")
        return None
    
    if not commit_sha:
        print(f"   ❌ No commit SHA provided")
        return None
    
    try:
        print(f"   🔍 Fetching parent commit for {commit_sha[:8]}...")
        url = f"https://api.github.com/repos/{repo_name}/commits/{commit_sha}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        response = requests.get(url, headers=headers)
        print(f"   📡 Parent commit API status: {response.status_code}")
        
        if response.status_code == 200:
            commit_data = response.json()
            parents = commit_data.get("parents", [])
            
            if parents:
                parent_sha = parents[0]["sha"]
                print(f"   ✅ Found parent commit: {parent_sha[:8]}")
                return parent_sha
            else:
                print(f"   ⚠️  No parent commits - this is likely the initial commit")
                return None
        else:
            print(f"   ❌ Failed to get parent commit: {response.status_code} - {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"   ❌ Error getting parent commit: {str(e)}")
        return None

def determine_file_status(file_info: dict, commits: list, file_path: str) -> str:
    """Determine the actual status of the file (added, modified, removed)"""
    for commit in commits:
        if file_path in commit.get("added", []):
            return "added"
        elif file_path in commit.get("removed", []):
            return "removed"
    
    if file_info["before_content"] and not file_info["after_content"]:
        return "removed"
    elif not file_info["before_content"] and file_info["after_content"]:
        return "added"
    elif file_info["before_content"] and file_info["after_content"]:
        return "modified"
    
    return "unknown"

async def analyze_changes_with_gemini(file_changes: list, repo_name: str, commit_message: str) -> dict:
    """Use Gemini AI to analyze the code changes and provide insights"""
    if not GEMINI_API_KEY:
        print("🤖 Gemini API key not configured - skipping AI analysis")
        return {"ai_analysis": "Gemini not configured"}
    
    try:
        # Prepare the context for Gemini
        changed_files_summary = []
        for file in file_changes:
            file_summary = {
                "file": file["file_path"],
                "status": file["status"],
                "changes": f"+{file.get('after_lines', 0) - file.get('before_lines', 0)} lines",
                "size_change": f"+{file.get('after_size', 0) - file.get('before_size', 0)} chars"
            }
            changed_files_summary.append(file_summary)
        
        # Get actual content samples for analysis
        content_samples = []
        for file in file_changes:
            if file["status"] == "modified" and file.get("before_content") and file.get("after_content"):
                # Show a diff-like preview
                before_preview = file["before_content"][:200] + "..." if len(file["before_content"]) > 200 else file["before_content"]
                after_preview = file["after_content"][:200] + "..." if len(file["after_content"]) > 200 else file["after_content"]
                
                content_samples.append(f"""
File: {file['file_path']}
Before: {before_preview}
After: {after_preview}
""")
        
        context = f"""
Repository: {repo_name}
Commit Message: {commit_message}

Files Changed:
{json.dumps(changed_files_summary, indent=2)}

Content Changes:
{''.join(content_samples) if content_samples else 'No content changes available'}
"""
        
        # Create Gemini prompt
        prompt = f"""
You are a senior software engineer reviewing GitHub code changes. Analyze these changes and provide:

1. **Change Summary**: Brief overview of what was modified
2. **Impact Assessment**: Potential effects on the codebase
3. **Quality Check**: Any obvious issues or improvements needed
4. **Security Notes**: If applicable, any security considerations
5. **Suggestions**: Actionable recommendations

Keep it concise and focused on the actual changes shown.

Here are the changes:
{context}

Please provide your analysis in a structured but concise format.
"""
        
        print("🤖 Sending changes to Gemini for analysis...")
        
        # Initialize the Gemini model
        model = genai.GenerativeModel('gemini-pro')
        
        response = model.generate_content(prompt)
        
        ai_analysis = response.text
        
        print("✅ Gemini analysis completed!")
        
        return {
            "ai_analysis": ai_analysis,
            "model_used": "gemini-pro",
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Gemini analysis error: {str(e)}")
        return {"ai_analysis": f"Analysis failed: {str(e)}"}
    


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting GitHub AI Agent Server...")
    print(f"🔧 GITHUB_TOKEN configured: {bool(GITHUB_TOKEN)}")
    print(f"🔧 WEBHOOK_SECRET configured: {bool(WEBHOOK_SECRET)}")
    uvicorn.run(app, host="localhost", port=8000, log_level="info")

