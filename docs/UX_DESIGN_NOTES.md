# UX Design Notes - Decision Support System

## 🎯 Core Principle

**Every screen answers: "What should I do NOW?" or "Why should I act?"**

If a component doesn't answer these questions, it was removed.

## 📱 Screen-by-Screen UX Decisions

### 🏠 Decision Dashboard (Landing Page)

**Purpose:** Immediate action in under 3 minutes

**Design Decisions:**

1. **Summary Cards (Top)**
   - **Why:** Manager needs instant overview
   - **Design:** 4 cards, color-coded by urgency
   - **Language:** "Critical Expiry" not "High Risk Score"
   - **Numbers:** Large, bold - easy to scan

2. **Top Actions (Below Cards)**
   - **Why:** Manager needs clear priorities
   - **Limit:** MAX 5 actions (cognitive load)
   - **Priority Order:** Critical expiry → Understock → Overstock
   - **Each Action Shows:**
     - Action type (DISCOUNT, REORDER, etc.)
     - Item name (not just ID)
     - Reason (plain language)
     - Impact (business benefit)
     - Specific details (discount %, quantity)

3. **Action Buttons**
   - **Why:** Manager needs to track decisions
   - **Design:** "Mark as Done" and "Ignore for Today"
   - **Feedback:** Immediate confirmation
   - **State:** Persists in session

**UX Rationale:**
- Manager opens app → sees 4 numbers → sees 5 actions → acts → done
- No scrolling needed for critical decisions
- Color coding (red/yellow/blue) = instant priority recognition

### 📦 Inventory Risk View

**Purpose:** Risk awareness and filtering

**Design Decisions:**

1. **Table Layout**
   - **Why:** Manager needs to scan many items quickly
   - **Columns:** Item name, Risk level, Days since sale, Stock, Action
   - **No technical metrics:** Removed "risk_score", "std_dev", etc.

2. **Filters**
   - **Why:** Manager needs to focus on specific risks
   - **Options:** Critical Expiry, Need Reorder, Overstocked
   - **Language:** Business terms, not technical

3. **Search**
   - **Why:** Manager knows item names, not IDs
   - **Design:** Search by name or ID

**UX Rationale:**
- Manager thinks: "What's expiring?" → Filter → See list → Act
- Table format = familiar, scannable
- Filters = focus, not overwhelm

### 📈 Demand Insight

**Purpose:** Build confidence in decisions

**Design Decisions:**

1. **Simple Chart**
   - **Why:** Manager needs to see trend, not analyze
   - **Design:** Single line chart, no clutter
   - **No technical indicators:** Removed moving averages, confidence intervals

2. **Business Explanation Below Chart**
   - **Why:** Chart alone doesn't answer "So what?"
   - **Content:**
     - Average daily demand
     - Peak demand
     - Trend (increasing/stable/decreasing)
     - Recommendation in plain language

3. **Item Selector**
   - **Why:** Manager selects item they're planning for
   - **Design:** Dropdown, limited to 100 items (performance)

**UX Rationale:**
- Manager thinks: "How much should I prep?" → Select item → See chart → Read explanation → Know quantity
- Chart = visual confirmation
- Explanation = actionable insight

### 🧠 All Recommendations

**Purpose:** Complete transparency and trust

**Design Decisions:**

1. **Filterable List**
   - **Why:** Manager needs to see all options
   - **Filters:** By type, by priority
   - **Limit:** Show 20 at a time (performance)

2. **Action Cards**
   - **Why:** Each recommendation needs full context
   - **Shows:** Action, item, reason, impact, details
   - **Design:** Same card style as Top Actions (consistency)

**UX Rationale:**
- Manager thinks: "What else should I know?" → Filter → See all → Make informed decisions
- Transparency = trust
- Complete info = confidence

## 🎨 Visual Design Decisions

### Color System

- **Red (#dc3545):** Act NOW (critical/urgent)
- **Yellow (#ffc107):** Watch closely (high priority)
- **Blue (#17a2b8):** Monitor (medium priority)
- **Green (#28a745):** Safe (low risk)

**Rationale:** Universal color language, no learning curve

### Typography

- **Headers:** Large, bold (2rem) - immediate hierarchy
- **Item Names:** Medium, semi-bold (1.1rem) - easy to scan
- **Reasons:** Italic, gray - secondary info
- **Impact:** Bold, green - positive reinforcement

**Rationale:** Visual hierarchy guides eye to decisions

### Spacing

- **Cards:** Generous padding (1.5rem) - easy to read
- **Between Actions:** Clear separation - no confusion
- **Margins:** Consistent - professional feel

**Rationale:** Breathing room = less cognitive load

## 💬 Language Decisions

### Removed Technical Terms

- ❌ "Risk Score" → ✅ "Risk Level"
- ❌ "TTL" → ✅ "Days Since Sale"
- ❌ "QOH" → ✅ "Current Stock"
- ❌ "ADU" → ✅ "Daily Demand"
- ❌ "Safety Stock" → ✅ "Extra Stock"
- ❌ "ML Forecast" → ✅ "Demand Forecast"

### Business Language Used

- "Items need action NOW"
- "Risk of running out"
- "Too much inventory"
- "Reduce waste and save money"
- "Prevent stockouts"

**Rationale:** Manager speaks business, not tech

## 🔄 Interaction Design

### Action Tracking

- **Mark as Done:** Manager completes action
- **Ignore for Today:** Manager defers decision
- **State Persists:** Actions don't reappear in session

**Rationale:** Manager needs to track what they've handled

### Feedback

- **Immediate:** Success/info message on action
- **Visual:** Button state changes
- **Clear:** "Marked X as done" confirmation

**Rationale:** Manager needs confidence their action was recorded

## 📊 Data Handling

### Loading States

- **Spinner:** "Loading your inventory decisions..."
- **Empty States:** Clear messages, not errors
- **Missing Data:** Graceful degradation

**Rationale:** Manager shouldn't see technical errors

### Performance

- **Caching:** Data cached after first load
- **Limits:** Top 5 actions, 20 recommendations displayed
- **Lazy Loading:** Only load what's needed

**Rationale:** Fast = usable in daily operations

## ✅ Quality Checklist

Every screen was tested against:

- [ ] Does it answer "What should I do?"
- [ ] Can manager act in under 3 minutes?
- [ ] Is language plain English?
- [ ] Are priorities clear?
- [ ] Is impact explained?
- [ ] Can manager trust the recommendation?

## 🎯 Final Test

**If a manager opens this app:**

1. ✅ They immediately see what's urgent (red cards)
2. ✅ They know what to do (Top 5 Actions)
3. ✅ They understand why (reasons in plain language)
4. ✅ They trust the recommendations (transparency)
5. ✅ They can act quickly (clear actions, buttons)

**Result:** Decision-support product, not analytics dashboard ✅
