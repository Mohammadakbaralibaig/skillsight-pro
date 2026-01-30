# **⚡ SkillSight Pro – AI + LLM Powered Resume Intelligence Platform**  
## **End-to-End DevOps Deployment with AWS, Docker, Kubernetes, Splunk & Automation**

SkillSight Pro is an advanced **AI-powered, LLM-based Resume & Career Analysis System** that evaluates resumes, extracts skills, matches job roles, and generates personalized insights using Machine Learning, Deep Learning, and modern Generative AI models.

This repository includes the **full AI application** along with **DevOps deployment automation** using AWS EC2, Docker, Kubernetes, Terraform, Ansible, and monitoring integration with Splunk.

 🚀 Key Features

 🤖 AI / ML / LLM Capabilities
- Named Entity Recognition (NER) for skill extraction  
- Embedding-based similarity scoring  
- LLM-generated job matching and insights  
- Resume scoring and ranking  
- Deep Learning text understanding  
- AI-generated career roadmap suggestions

 🧠 LLM Integration
Powered by:
- OpenAI GPT models  
- Mistral / LLaMA models  
- Transformer-based pipelines  
- Contextual embeddings  

LLM outputs include:
- Resume summary  
- Job role mapping  
- Skill gap analysis  
- Improvement recommendations  
- Interview preparation insights  

🛠 DevOps & Cloud Architecture

SkillSight Pro demonstrates a complete real-world DevOps workflow:

## ☁️ AWS Infrastructure
- EC2 (Compute)  
- Security Groups  
- IAM  
- Terraform automation  
- SSH Secure Access  
- Public deployment  

## 🔧 Configuration Management – Ansible
Automates:
- System update  
- Python & dependency installation  
- Cloning GitHub repo  
- Application startup  
- Docker installation (optional)

## 🐳 Containerization – Docker
Includes:
- Dockerfile  
- Build & run instructions  
- Portable container support  

## ☸️ Orchestration – Kubernetes
Includes:
- `deployment.yaml` — scalable deployment  
- `service.yaml` — LoadBalancer/ClusterIP  
- Ready for production clusters  

## 📊 Monitoring – Splunk Integration
Supports:
- Log forwarding  
- HTTP Event Collector (HEC)  
- Error + request monitoring  
- Backend log insights  


# 📁 Project Structure

skillsight-pro-ai-llm-devops/
│
├── app.py 
├── dl_ner.py 
├── dl_embeddings.py 
├── Dockerfile 
├── deployment.yaml 
├── service.yaml 
├── requirements.txt 
└── .env 


# 🧠 AI/LLM Pipeline

## ✔ Skill Extraction (NER)
Model:
tner/bertweet-base-twitterner2021

yaml
Copy code

## ✔ Embedding Layer
Used for:
- Job-role similarity  
- Skill matching  
- Ranking and scoring  

## ✔ LLM Intelligence
LLMs generate:
- Feedback  
- Recommendations  
- Summaries  
- Job match results  







# 🐳 Docker Deployment

### Build:
docker build -t skillsight-pro .



### Run:
docker run -p 8501:8501 skillsight-pro

# ☸️ Kubernetes Deployment

kubectl apply -f deployment.yaml
kubectl apply -f service.yaml



# 📊 Splunk Monitoring Setup

1. Enable **HTTP Event Collector (HEC)**  
2. Add Splunk HEC token + URL into `.env`  
3. Application sends logs using Python logging  



# 🏁 Conclusion

SkillSight Pro is a complete **AI + ML + LLM + DevOps** platform demonstrating:

- Cloud Infrastructure  
- Automation  
- Containerization  
- Orchestration  
- AI engineering  
- Production-grade architecture  


# 👤 Author

Venkata Varshith Narayanam 
B.Tech CSE – Data Science & Machine Learning  
DevOps • AI Engineering • Cloud • MLOps -- LLM -- Deep learning
