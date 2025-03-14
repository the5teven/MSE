# **📌 SME Library - README**  

This **README** provides an overview of the **SME (Simulation-Based Model Estimation) Library**, its **file structure**, how to **install and use** it, and key features like **active learning, FAISS search, and dynamic simulation-based training**.  

---

## **📂 File Structure**  
Here is the **organized file structure** of the SME library:

```
📦 SME
├── 📁 sme                 # Core SME Library
│   ├── __init__.py        # Initializes the library
│   ├── config.py          # Defines SMEConfig (library settings)
│   ├── models.py          # Defines SMEModel (main model)
│   ├── dataset.py         # SimulationDataset class
│   ├── losses.py          # Custom loss functions (CompositeLoss, MemoryBankLoss)
│   ├── optim.py           # Optimizers (EMA, Curriculum Learning)
│   ├── simulator.py       # GeneralSimulator (time series generation)
│   ├── utils.py           # Helper functions
│
├── 📁 examples            # Example scripts
│   ├── test.py            # Fully tests SMEModel (dynamic training & estimation)
│   ├── quickstart.py      # Minimal working example
│
├── 📁 docs                # Documentation
│   ├── README.md          # This file
│   ├── usage.md           # Detailed usage guide
│   ├── config.md          # Configuration options
│
├── setup.py               # Installation script
├── requirements.txt       # Dependencies
└── .gitignore             # Files to ignore in Git
```

---

## **🛠️ Installation**
To install the SME library, first **clone the repository** and **install dependencies**:

```bash
git clone https://github.com/your-repo/SME.git
cd SME
pip install -r requirements.txt
```

Dependencies include:
- `torch`
- `faiss-cpu` (or `faiss-gpu` for CUDA users)
- `numpy`
- `matplotlib`
- `tqdm`

---

## **🚀 Quick Start**
This **minimal example** shows how to **simulate data, train SMEModel, and estimate parameters**.

```python
import torch
from sme import SMEModel, SimulatorConfig, GeneralSimulator, SMEConfig

# ✅ Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define Configuration
config = SMEConfig(
    encoder_class=Encoder,  
    emulator_class=Emulator,  
    use_pretraining=True,
    use_active_learning=True,
    batch_size=64,
    num_epochs=100,
    learning_rate=5e-4,
)

# ✅ Initialize Model
sme_model = SMEModel(config)

# ✅ Train Model (dynamically generates training data)
sme_model.train()

# ✅ Simulate Test Data
sim_config = SimulatorConfig(model_type="VAR", params={"phi": torch.rand((2, 2))}, T=100, n_vars=2)
Y_star = GeneralSimulator(sim_config).simulate().cpu().numpy()

# ✅ Estimate Parameters
phi_est = sme_model.estimate_phi(Y_star, candidate_pool=torch.rand((500, 5)))
print(f"Estimated Parameters: {phi_est}")
```

---

## **📌 Key Features**
| **Feature** | **Description** |
|------------|----------------|
| ✅ **Dynamic Data Generation** | Training data is **generated on-the-fly**, reducing memory usage |
| ✅ **Active Learning** | Uses **uncertainty-based selection** to choose the most **informative samples** |
| ✅ **Break Point Estimation (`tau`)** | Uses **FAISS nearest neighbors** for fast estimation |
| ✅ **Pretraining Support** | Pretraining on simpler tasks **before full training** |
| ✅ **FAISS Integration** | Uses **GPU-optimized FAISS search** for nearest neighbors |
| ✅ **Memory Bank for Contrastive Learning** | Supports **feature memory banks** for learning representations |
| ✅ **Fully GPU-Optimized** | Supports **CUDA acceleration** for training & FAISS search |

---

## **🛠️ Configuration (`SMEConfig`)**
`SMEConfig` **controls how the library behaves**.

```python
config = SMEConfig(
    encoder_class=Encoder,  
    emulator_class=Emulator,  
    input_dim=(100, 2), 
    use_pretraining=True,
    use_active_learning=True,  
    candidate_pool_size=500, 
    training_steps_per_epoch=100,  
    pretraining_samples=1000,  
    pretraining_model="VAR",  
    refinement_steps=10,
    refinement_lr=0.001,
    use_memory_bank_updates=True,
    use_early_stopping=True,
    batch_size=64,
    num_epochs=200,
    learning_rate=5e-4,
    early_stopping_patience=20
)
```

---

## **🧪 Running Tests**
To **fully test the SME library**, run:

```bash
python examples/test.py
```

This will:
✅ **Train the SME model dynamically**  
✅ **Estimate parameters on new simulated data**  
✅ **Measure the accuracy of tau estimation**  
✅ **Generate evaluation plots**  

---

## **📊 Results & Evaluation**
After running `test.py`, you'll see **evaluation metrics** like:

```bash
Mean Absolute Error in tau: 4.73
```

And **graphs** showing **tau estimation error distributions**.

---

## **📞 Contact & Support**
For issues, open a GitHub issue or contact `your-email@example.com`.

---

### **🚀 Now You're Ready to Use SME!**  
✅ **Train models dynamically**  
✅ **Estimate parameters efficiently**  
✅ **Leverage active learning**  
