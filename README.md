# ChatPartner — Personal AI Companion

A personal AI companion application engineered with a clean decoupled architecture:
- **Frontend**: React 19, Vite, Tailwind CSS v4
- **Backend**: Python, Django 4.2+, Django REST Framework (DRF)
- **Database**: PostgreSQL
- **AI Engine**: OpenAI API (`gpt-4o-mini`)
- **Future Capabilities**: Long-term memory, proactive scheduled check-ins, push notifications, and voice conversations.

---

## Project Structure

```
chatpartner/
├── backend/                  # Django REST Framework backend
│   ├── config/               # Project configuration, URLs, settings, health checks
│   │   ├── settings.py       # DRF, PostgreSQL, CORS, OpenAI configuration
│   │   ├── urls.py           # API route mappings (/api/health/, /api/chat/, /api/conversations/, /api/memories/)
│   │   └── views.py          # Diagnostic and health check views
│   ├── chat/                 # AI Companion chat, conversation persistence & long-term memory
│   │   ├── models.py         # Conversation, ChatMessage, CompanionMemory models
│   │   ├── services.py       # CompanionChatService with memory prompt injection
│   │   ├── views.py          # Chat, conversation, and memory CRUD endpoints
│   │   └── tests.py          # Persistence, isolation, and memory tests
│   ├── proactive_messaging/  # Companion proactive messaging domain (preserved foundation)
│   ├── manage.py             # Administrative CLI
│   ├── requirements.txt      # Python dependencies
│   ├── .env.example          # Backend environment variables template
│   └── .env                  # Local backend environment variables (git-ignored)

├── frontend/                 # React + Vite frontend
│   ├── src/
│   │   ├── services/api.js   # API client service and health check fetcher
│   │   ├── App.jsx           # Diagnostic and platform foundation dashboard
│   │   ├── main.jsx          # React DOM entry point
│   │   └── index.css         # Tailwind CSS v4 styling
│   ├── vite.config.js        # Vite + Tailwind v4 config, dev server on port 5173
│   ├── package.json          # Node dependencies
│   ├── .env.example          # Frontend environment variables template
│   └── .env                  # Local frontend environment variables (git-ignored)
├── .gitignore                # Git ignore rules for Python, Node, and secrets
└── README.md                 # Setup and architecture documentation
```

---

## Getting Started

### 1. Backend Setup (Django & PostgreSQL)

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to configure your PostgreSQL credentials:
   ```ini
   DEBUG=True
   SECRET_KEY=your-secret-key-here
   DB_NAME=chatpartner_db
   DB_USER=your_postgres_user
   DB_PASSWORD=your_postgres_password
   DB_HOST=127.0.0.1
   DB_PORT=5432
   CORS_ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Install Python dependencies:**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

4. **Run migrations:**
   ```bash
   python3 manage.py migrate
   ```

5. **Start the Django development server:**
   ```bash
   python3 manage.py runserver 127.0.0.1:8000
   ```
   The health check API will be available at `http://127.0.0.1:8000/api/health/`.

---

### 2. Frontend Setup (React + Vite + Tailwind CSS)

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Ensure `VITE_API_BASE_URL` points to your Django API:
   ```ini
   VITE_API_BASE_URL=http://127.0.0.1:8000/api
   ```

3. **Install dependencies:**
   ```bash
   npm install
   ```

4. **Start the Vite development server:**
   ```bash
   npm run dev
   ```
   Access the frontend application at `http://localhost:5173`.

---

## Health Check & Verification

### Backend Verification:
```bash
cd backend
python3 manage.py test
```
All 25 tests should pass, covering:
- Health check endpoint (`/api/health/` and `/api/health-check/`)
- CORS response headers for `http://localhost:5173`
- Database connectivity verification
- Proactive messaging foundation and worker models

### Frontend Verification:
```bash
cd frontend
npm run build
```
Builds production-ready bundles using Vite and Tailwind CSS.
