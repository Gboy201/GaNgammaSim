# Physics Simulation Web Application

This is a modern web application for running and visualizing physics simulations. It features a React frontend with a beautiful UI and a FastAPI backend that integrates with the physics simulation engine.

## Project Structure

```
web-app/
├── frontend/          # React frontend application
│   ├── src/
│   │   ├── components/
│   │   └── App.tsx
│   ├── package.json
│   └── vite.config.ts
└── backend/          # FastAPI backend server
    ├── main.py
    └── requirements.txt
```

## Setup Instructions

### Backend Setup

1. Create a Python virtual environment:

```bash
cd web-app/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the backend server:

```bash
python main.py
```

The backend will be available at http://localhost:8000

### Frontend Setup

1. Install Node.js dependencies:

```bash
cd web-app/frontend
npm install
```

2. Start the development server:

```bash
npm start
```

The frontend will be available at http://localhost:3000

## Features

- Modern, responsive UI built with Chakra UI
- Real-time data visualization with Recharts
- Interactive form for simulation parameters
- FastAPI backend for efficient simulation processing
- Automatic data refresh during simulation
- Professional charts and graphs for result visualization

## API Endpoints

- `POST /api/simulate`: Start a new simulation with provided parameters
- `GET /api/results`: Get current simulation results

## Development

- Frontend is built with React 18, TypeScript, and Vite
- Backend uses FastAPI with async support
- Real-time data updates using React Query
- Modular component architecture for easy maintenance
