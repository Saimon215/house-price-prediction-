import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Annual appreciation rate based on historical US housing market average
ANNUAL_APPRECIATION_RATE = 0.04  # 4% per year

# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

features = ['beds', 'baths', 'size', 'lot_size', 'zip_code']

X_train = train[features]
y_train = train['price']
X_test  = test[features]
y_test  = test['price']

# ──────────────────────────────────────────────
# 2. Train model
# ──────────────────────────────────────────────
print("\nTraining model...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

test_preds = model.predict(X_test)
mae = mean_absolute_error(y_test, test_preds)
r2  = r2_score(y_test, test_preds)

print("\n" + "=" * 44)
print("          MODEL PERFORMANCE")
print("=" * 44)
print(f"  Mean Absolute Error : ${mae:>14,.2f}")
print(f"  R² Score            :  {r2:>13.4f}")
print("=" * 44)

# ──────────────────────────────────────────────
# 3. Get house details from user
# ──────────────────────────────────────────────
print("\n" + "=" * 44)
print("       ENTER YOUR HOUSE DETAILS")
print("=" * 44)

while True:
    try:
        beds     = int(input("  Bedrooms              : "))
        baths    = float(input("  Bathrooms             : "))
        size     = float(input("  House size (sqft)     : "))
        lot_size = float(input("  Lot size (sqft)       : "))
        zip_code = int(input("  Zip code              : "))
        if beds <= 0 or baths <= 0 or size <= 0 or lot_size <= 0:
            print("  [!] All values must be greater than zero. Try again.\n")
            continue
        break
    except ValueError:
        print("  [!] Invalid input — please enter numbers only.\n")

# ──────────────────────────────────────────────
# 4. Predict current price
# ──────────────────────────────────────────────
user_house = pd.DataFrame([{
    'beds': beds, 'baths': baths,
    'size': size, 'lot_size': lot_size,
    'zip_code': zip_code
}])

current_price = model.predict(user_house)[0]

# ──────────────────────────────────────────────
# 5. Build 10-year projection
# ──────────────────────────────────────────────
base_year = datetime.datetime.now().year
years     = list(range(base_year, base_year + 11))
prices    = [current_price * ((1 + ANNUAL_APPRECIATION_RATE) ** i) for i in range(11)]

# ──────────────────────────────────────────────
# 6. Print projection table
# ──────────────────────────────────────────────
print("\n" + "=" * 44)
print("       10-YEAR PRICE PROJECTION")
print("=" * 44)
print(f"  {'Year':<8} {'Projected Price':>16}  {'Growth':>8}")
print("  " + "-" * 38)
for i, (yr, pr) in enumerate(zip(years, prices)):
    tag    = "  <- Current" if i == 0 else ""
    growth = f"+{((pr / current_price - 1) * 100):.1f}%" if i > 0 else "  baseline"
    print(f"  {yr:<8} ${pr:>15,.2f}  {growth:>8}{tag}")
print("=" * 44)
print(f"  Appreciation rate used: {ANNUAL_APPRECIATION_RATE * 100:.1f}% / year")
print("=" * 44)

# ──────────────────────────────────────────────
# 7. Graphs
# ──────────────────────────────────────────────
dollar_fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("House Price Analysis", fontsize=16, fontweight='bold', y=1.01)

# --- Left: Actual vs Predicted ---
ax1 = axes[0]
ax1.scatter(y_test, test_preds, alpha=0.55, color='steelblue',
            edgecolors='white', linewidths=0.4, s=60)
lo = min(y_test.min(), test_preds.min())
hi = max(y_test.max(), test_preds.max())
ax1.plot([lo, hi], [lo, hi], 'r--', linewidth=1.8, label='Perfect fit')
ax1.set_xlabel("Actual Price", fontsize=11)
ax1.set_ylabel("Predicted Price", fontsize=11)
ax1.set_title("Model Accuracy\n(Actual vs Predicted)", fontsize=12)
ax1.xaxis.set_major_formatter(dollar_fmt)
ax1.yaxis.set_major_formatter(dollar_fmt)
ax1.tick_params(axis='x', rotation=30)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.25)

# Add R² annotation
ax1.annotate(f"R² = {r2:.3f}\nMAE = ${mae:,.0f}",
             xy=(0.05, 0.92), xycoords='axes fraction',
             fontsize=10, color='navy',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# --- Right: 10-year projection ---
ax2 = axes[1]
ax2.plot(years, prices, 'o-', color='seagreen', linewidth=2.5,
         markersize=8, markerfacecolor='white', markeredgewidth=2.2)
ax2.fill_between(years, prices, alpha=0.12, color='seagreen')

# Annotate each point
for yr, pr in zip(years, prices):
    ax2.annotate(f"${pr:,.0f}",
                 xy=(yr, pr), xytext=(0, 10),
                 textcoords='offset points', ha='center',
                 fontsize=7.5, color='darkgreen')

ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("Projected Price", fontsize=11)
ax2.set_title(
    f"10-Year Price Projection\n"
    f"{beds}bd / {baths}ba  |  {size:,.0f} sqft  |  ZIP {zip_code}",
    fontsize=12
)
ax2.yaxis.set_major_formatter(dollar_fmt)
ax2.set_xticks(years)
ax2.tick_params(axis='x', rotation=30)
ax2.grid(True, alpha=0.25)
ax2.annotate(f"{ANNUAL_APPRECIATION_RATE*100:.0f}% annual appreciation assumed",
             xy=(0.03, 0.94), xycoords='axes fraction',
             fontsize=9, color='gray',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()
