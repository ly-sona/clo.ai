# clo.ai

VLSI Layout Optimizer with XGBoost + DEAP

## 🧠 Project Overview

**clo.ai** is a machine learning-driven VLSI layout optimization system that combines XGBoost regression with evolutionary algorithms to reduce power consumption and improve timing in integrated circuits. The system analyzes circuit netlists, predicts performance metrics, and automatically adjusts gate sizes to achieve an optimal balance between power and delay.

## ⚡ Why Faster = Greener

Delay optimization helps reduce power consumption in three complementary ways:

- **Lower Switching Activity**: Fewer glitches → fewer unnecessary transitions → less dynamic power (P ∝ CV²f)
- **Slack Turns Into Voltage-Headroom**: Shorter paths allow lower voltage or extra functionality with quadratic power savings
- **Smaller, Cooler Gates**: Right-sized transistors reduce capacitive loading and leakage

## 🛠️ How It Works

Our layout optimization model follows a five-stage process:

1. **Parse & Sketch**: Read ISCAS-85 or CircuitNet bench files, discover inputs → gates → outputs, and assign (x,y) coordinates
2. **Simulate & Sample**: Run quick transistor-sizing sweeps to generate thousands of <delay, power> training pairs
3. **Learn the Physics**: Train an XGBoost regressor on gate count, fan-in, and placement density data
4. **Search for Better Layouts**: Use DEAP genetic algorithm to find optimal sizing weights for delay reduction
5. **Validate & Export**: Write optimized weights back into the netlist and verify with ground-truth simulation

## ⚙️ Tech Stack

- **Frontend**: React, TypeScript, Tailwind CSS, Framer Motion
- **Visualization**: D3.js for circuit visualization
- **Backend**: Python, Flask
- **ML Framework**: XGBoost, DEAP (Distributed Evolutionary Algorithms in Python)
- **Data Processing**: NumPy, Pandas
- **Circuit Parsing**: Custom ISCAS85 parser

## 🚀 Key Features

- Web-based interface for uploading and optimizing circuits
- Interactive visualization of circuit layouts and optimized results
- Drag-and-drop workflow for easy circuit file uploads
- Multi-objective optimization balancing power and performance
- Comparison view between original and optimized layouts

## 📊 Model Performance

The model is trained on 2,000,000 samples from the CircuitNet dataset, focusing on the IR drop features. Key model parameters:

- XGBoost: max_depth=3, learning_rate=0.1, early_stopping_rounds=50
- DEAP: ngen=10-20, sigma=0.2, mu=1.0
- Circuit simulation: gate_size_range=0.5-2.0

Achieved an average of 15-20% power reduction across benchmark circuits while maintaining or improving timing performance.

## 🏃‍♂️ Getting Started

### Prerequisites

- Node.js (v16+)
- Python (v3.8+)
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/ly-sona/clo.ai.git
   cd clo.ai
   ```

2. Install frontend dependencies:
   ```
   cd frontend/application
   npm install
   ```

3. Install backend dependencies:
   ```
   cd ../../backend
   pip install -r requirements.txt
   ```

4. Start the development servers:
   ```
   # Terminal 1 (Frontend)
   cd frontend/application
   npm run dev
   
   # Terminal 2 (Backend)
   cd backend
   python app.py
   ```

5. Open your browser to http://localhost:5173

## 📁 Directory Structure

```
clo.ai/
├── backend/             # Python backend and API
├── circuit_files/       # Example and test circuit files
├── frontend/            # React frontend application
│   └── application/     # Main frontend code
├── layouts/             # Generated layout files
├── models/              # ML models and training scripts
│   ├── utils/           # Helper utilities
│   └── old scripts/     # Legacy scripts for reference
└── static/              # Static assets
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

- Mentor: Anu Boyapati
- Mentees: Ishita Saran, Patrick Sigler, Quan Dang, Orvin Ahmed, and Adwaith Moothezhath
