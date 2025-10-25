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
import google.api_core.exceptions

load_dotenv()

app = FastAPI(
    title="GitHub AI Agent",
    description="AI-powered GitHub automation agent",
    version="1.0.0"
)

WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if GEMINI_API_KEY:
    try:
        genai.configure(
            api_key=GEMINI_API_KEY,
            transport='rest',
        )
        print(f"🔮 Gemini AI configured successfully!")
        
        # try:
        #     models = genai.list_models()
        #     available_models = [model.name for model in models]
        #     print(f"📋 Available Gemini models: {available_models}")
        # except Exception as e:
        #     print(f"⚠️  Could not list models: {e}")
            
    except Exception as e:
        print(f"❌ Gemini configuration failed: {e}")
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
        url = f"{self.base_url}/repos/{repo}/contents/{file_path}?ref={ref}"
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return None
            elif response.status_code == 403:
                return None
            else:
                return None
        except Exception as e:
            return None

github_api = GitHubAPI(GITHUB_TOKEN)

@app.get("/")
async def root():
    return {"message": "GitHub AI Agent is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "github-ai-agent"}

def verify_webhook_signature(payload: bytes, signature: str) -> bool:
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
    try:
        payload = await request.body()
        payload_str = payload.decode('utf-8')
        
        if WEBHOOK_SECRET and x_hub_signature_256:
            if not verify_webhook_signature(payload, x_hub_signature_256):
                raise HTTPException(status_code=401, detail="Invalid webhook signature")
        
        json_payload = json.loads(payload_str)
        
        print(f"📨 Received GitHub event: {x_github_event}")
        
        sender = json_payload.get("sender", {})
        if sender.get("type") == "Bot" or "bot" in sender.get("login", "").lower():
            print("🤖 Ignoring bot-generated push event")
            return JSONResponse(
                status_code=200,
                content={"message": "Bot event ignored to prevent loops"}
            )
        
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
    repo_name = payload["repository"]["full_name"]
    commits = payload.get("commits", [])
    
    print(f"🚀 Push event for {repo_name}")
    print(f"📊 Commits: {len(commits)}")
    
    all_changed_files = set()
    for commit in commits:
        all_changed_files.update(commit.get("added", []))
        all_changed_files.update(commit.get("modified", []))
        all_changed_files.update(commit.get("removed", []))
    
    changed_files = [f for f in list(all_changed_files) if f.lower() != 'readme.md']
    
    print(f"📄 Changed files (excluding README.md): {changed_files}")
    
    if not changed_files:
        print("📝 Only README.md changed or no files changed - skipping analysis")
        return {
            "status": "skipped",
            "event": "push", 
            "repo": repo_name,
            "reason": "Only README.md changed or no files changed",
            "commit_count": len(commits)
        }
    
    file_changes_analysis = await analyze_file_changes(repo_name, commits, changed_files)
    
    ai_analysis_result = {}
    if commits and GEMINI_API_KEY and file_changes_analysis:
        commit_message = commits[-1].get("message", "No commit message")
        ai_analysis_result = await analyze_changes_with_gemini(
            file_changes_analysis, 
            repo_name, 
            commit_message
        )
        
        if (ai_analysis_result.get("ai_analysis") and 
            not ai_analysis_result.get("ai_analysis", "").startswith("Analysis failed") and
            not ai_analysis_result.get("ai_analysis", "").startswith("Gemini not configured")):
            
            current_commit_sha = commits[-1]["id"]
            await update_readme_with_analysis(
                repo_name, 
                ai_analysis_result["ai_analysis"], 
                current_commit_sha,
                GITHUB_TOKEN
            )
    else:
        ai_analysis_result = {"ai_analysis": "No analysis performed"}
    
    return {
        "status": "processed",
        "event": "push",
        "repo": repo_name,
        "commit_count": len(commits),
        "files_analyzed": changed_files,
        "file_changes": file_changes_analysis,
        "ai_analysis": ai_analysis_result
    }

async def handle_pull_request_event(payload: dict):
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
            current_commit_sha = commits[-1]["id"]
            parent_commit_sha = commits[0].get("parents", [{}])[0].get("sha") if commits[0].get("parents") else None
            
            print(f"   🔄 Current commit: {current_commit_sha[:8]}")
            print(f"   🔄 Parent commit: {parent_commit_sha[:8] if parent_commit_sha else 'None'}")
            
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
                        print(f"   ⚠️  No BEFORE content found")
                else:
                    print(f"   ⚠️  No parent commit")
            
            print(f"   🔍 Getting AFTER content from current: {current_commit_sha[:8]}")
            after_content = await get_file_content_at_commit(repo_name, file_path, current_commit_sha)
            if after_content:
                file_info["after_content"] = after_content["content"]
                file_info["after_size"] = after_content["size"]
                file_info["after_lines"] = after_content["lines"]
                print(f"   ✅ AFTER content: {file_info['after_size']} chars, {file_info['after_lines']} lines")
            else:
                print(f"   ⚠️  No AFTER content found")
            
            for commit in commits:
                if file_path in commit.get("added", []):
                    file_info["status"] = "added"
                    print(f"   📝 Status: ADDED")
                    break
                elif file_path in commit.get("removed", []):
                    file_info["status"] = "removed" 
                    print(f"   📝 Status: REMOVED")
                    break
                elif file_path in commit.get("modified", []):
                    file_info["status"] = "modified"
                    print(f"   📝 Status: MODIFIED")
                    break
            
            if file_info["status"] == "modified":
                if not file_info["before_content"] and file_info["after_content"]:
                    file_info["status"] = "added"
                    print(f"   📝 Status: ADDED")
                elif file_info["before_content"] and not file_info["after_content"]:
                    file_info["status"] = "removed"
                    print(f"   📝 Status: REMOVED")
        
        print(f"   📊 FINAL CONTENT SUMMARY:")
        print(f"     - Status: {file_info['status']}")
        print(f"     - Before: {file_info['before_size']} chars, {file_info['before_lines']} lines")
        print(f"     - After: {file_info['after_size']} chars, {file_info['after_lines']} lines")
        
        if file_info["before_content"] and file_info["after_content"]:
            content_diff = len(file_info['after_content']) - len(file_info['before_content'])
            print(f"     - Size change: {content_diff:+d} chars")
        
        file_analysis.append(file_info)
    
    print(f"\n✅ File content analysis complete for {len(file_analysis)} files")
    return file_analysis

async def get_file_content_at_commit(repo_name: str, file_path: str, commit_sha: str) -> dict:
    if not GITHUB_TOKEN or not commit_sha:
        return None
    
    try:
        file_data = github_api.get_file_content(repo_name, file_path, commit_sha)
        
        if file_data and "content" in file_data:
            content = base64.b64decode(file_data["content"]).decode('utf-8')
            
            return {
                "content": content,
                "size": len(content),
                "lines": content.count('\n') + 1,
                "encoding": file_data.get("encoding", "base64"),
                "sha": file_data.get("sha", "")
            }
    except Exception:
        pass
    
    return None

async def get_parent_commit(repo_name: str, commit_sha: str) -> str:
    if not GITHUB_TOKEN:
        return None
    
    try:
        url = f"https://api.github.com/repos/{repo_name}/commits/{commit_sha}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            commit_data = response.json()
            parents = commit_data.get("parents", [])
            
            if parents:
                parent_sha = parents[0]["sha"]
                return parent_sha
    except Exception:
        pass
    
    return None

async def analyze_changes_with_gemini(file_changes: list, repo_name: str, commit_message: str) -> dict:
    if not GEMINI_API_KEY:
        print("🤖 Gemini API key not configured - skipping AI analysis")
        return {"ai_analysis": "Gemini not configured"}
    
    try:
        changed_files_summary = []
        content_samples = []
        
        for file in file_changes:
            file_summary = {
                "file": file["file_path"],
                "status": file["status"],
                "changes": f"+{file.get('after_lines', 0) - file.get('before_lines', 0)} lines",
                "size_change": f"+{file.get('after_size', 0) - file.get('before_size', 0)} chars"
            }
            changed_files_summary.append(file_summary)
            
            if file["status"] == "modified" and file.get("before_content") and file.get("after_content"):
                before_preview = file["before_content"][:200] + "..." if len(file["before_content"]) > 200 else file["before_content"]
                after_preview = file["after_content"][:200] + "..." if len(file["after_content"]) > 200 else file["after_content"]
                
                content_samples.append(f"""
File: {file['file_path']}
Before: {before_preview}
After: {after_preview}
""")
            elif file["status"] == "added" and file.get("after_content"):
                content_preview = file["after_content"][:300] + "..." if len(file["after_content"]) > 300 else file["after_content"]
                content_samples.append(f"""
File: {file['file_path']} (NEWLY ADDED)
Content: {content_preview}
""")

        context = f"""
Repository: {repo_name}
Commit Message: {commit_message}

Files Changed:
{json.dumps(changed_files_summary, indent=2)}

Content Changes:
{''.join(content_samples) if content_samples else 'No content changes available'}
"""
        
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
        
        model_names_to_try = [
            'models/gemini-2.0-flash',
            'models/gemini-pro-latest',
            'models/gemini-2.0-flash-001',
        ]
        
        response = None
        last_error = None
        
        for model_name in model_names_to_try:
            try:
                print(f"   🔧 Trying model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                print(f"   ✅ Success with model: {model_name}")
                break
            except Exception as e:
                last_error = e
                continue
        
        if response is None:
            print(f"❌ All model attempts failed. Last error: {last_error}")
            return {"ai_analysis": f"Gemini analysis failed: {str(last_error)}"}
        
        ai_analysis = response.text
        
        print("✅ Gemini analysis completed!")
        print(f"📝 Analysis preview: {ai_analysis[:200]}...")
        
        return {
            "ai_analysis": ai_analysis,
            "model_used": model_name,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Gemini analysis error: {str(e)}")
        return {"ai_analysis": f"Analysis failed: {str(e)}"}

async def update_readme_with_analysis(repo_name: str, analysis: str, commit_sha: str, github_token: str):
    try:
        print(f"📝 Updating README.md with analysis for commit {commit_sha[:8]}...")
        
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        readme_url = f"https://api.github.com/repos/{repo_name}/contents/README.md"
        response = requests.get(readme_url, headers=headers)
        
        if response.status_code != 200:
            print(f"❌ Could not fetch README: {response.status_code}")
            return False
        
        readme_data = response.json()
        current_content = base64.b64decode(readme_data["content"]).decode('utf-8')
        readme_sha = readme_data["sha"]
        
        analysis_section = f"""
## 🤖 AI Code Analysis

**Last Analysis:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Commit:** `{commit_sha[:8]}`

### Analysis Report:
{analysis}

---
*Generated by GitHub AI Agent*
"""
        
        if "## 🤖 AI Code Analysis" in current_content:
            parts = current_content.split("## 🤖 AI Code Analysis")
            if len(parts) > 1:
                remaining = parts[1].split("---\n*Generated by GitHub AI Agent*")
                if len(remaining) > 1:
                    new_content = parts[0] + analysis_section + remaining[1]
                else:
                    new_content = parts[0] + analysis_section
            else:
                new_content = current_content + "\n" + analysis_section
        else:
            new_content = current_content + "\n" + analysis_section
        
        update_data = {
            "message": f"🤖 AI Analysis Update for {commit_sha[:8]}",
            "content": base64.b64encode(new_content.encode()).decode(),
            "sha": readme_sha,
            "branch": "main"
        }
        
        update_response = requests.put(readme_url, headers=headers, json=update_data)
        
        if update_response.status_code == 200:
            print("✅ README.md updated successfully with AI analysis!")
            return True
        else:
            print(f"❌ Failed to update README: {update_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating README: {str(e)}")
        return False

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting GitHub AI Agent Server...")
    print(f"🔧 GITHUB_TOKEN configured: {bool(GITHUB_TOKEN)}")
    print(f"🔧 WEBHOOK_SECRET configured: {bool(WEBHOOK_SECRET)}")
    uvicorn.run(app, host="localhost", port=8000, log_level="info")


