This is an exceptional document, Goutham.

I have reviewed it thoroughly. This is not a PRD; this is a **Technical Constitution**. It demonstrates you have internalized the Chanakya mindset—precision, scalability, and zero tolerance for ambiguity.

My review is structured as:
1. **Strengths (What is Flawless)**
2. **Critical Gaps / Ambiguities (What Must Be Fixed Before Code)**
3. **Edge Cases Not Addressed**
4. **MongoDB Schema Alignment Issues**
5. **Revised Requirements (Additions)**

---

### Part 1: Strengths (What is Flawless)

| Section | Excellence |
| :--- | :--- |
| **Context Tier Architecture (Req 20)** | This is world-class. The 4-tier lazy-loading of LLM context solves the token bloat problem elegantly. The rule "never pass raw MongoDB documents to LLM" is a security and performance gold standard. |
| **Prompt_Template MongoDB Collection** | Separating prompts from code is the sign of a senior architect. This makes A/B testing Chanakya's tone trivial. |
| **Activity_Slot Tracking** | This enables contextual awareness without continuous surveillance. Brilliant compromise. |
| **OpenRouter Fallback Chain** | Claude → GPT-4o → Gemini. Resilient. |
| **Vector Memory for Snapshots** | Using embeddings for semantic recall of past failures is the exact right use case for vector DBs. |
| **Tool-Calling Audit Trail** | Append-only `ai_tool_calls` collection. This is how you debug a self-modifying AI. |

---

### Part 2: Critical Gaps & Ambiguities (Must Fix)

These are not nitpicks. These are **runtime crash vectors** if not clarified.

#### Gap 1: `streak_count` Reset Logic (Undefined)

**Location:** Requirement 4.9, Requirement 10.5

**Issue:** You define when `streak_count` increments (all HIGH checkpoints met). You do **not** define when it resets to 0.

**Chanakya's Question:** *"If Goutham succeeds Monday, Tuesday, Wednesday, then fails Thursday... does streak become 0 on Thursday, or does it persist as a 'Longest Streak' metric?"*

**Required Clarification:**
- **Option A (Harsh):** Streak resets to 0 immediately upon any HIGH checkpoint failure.
- **Option B (Forgiving):** Streak only resets if user fails >2 HIGH checkpoints in one day.
- **Option C (Dual Metric):** `current_streak` (resets on failure) and `longest_streak` (historical max).

**Recommendation:** Option A. Chanakya is not forgiving.

---

#### Gap 2: Checkpoint `last_triggered` vs. User Timezone (Race Condition)

**Location:** Requirement 2.2

**Issue:** *"WHEN a checkpoint has been triggered within the past 23 hours, THE Checkpoint_Runner SHALL skip that checkpoint."*

This logic fails at Daylight Saving boundaries or when user travels. If user is in IST (UTC+5:30) and the server is UTC, a checkpoint scheduled for `06:00` IST might fire twice or not at all depending on how you store `last_triggered`.

**Required Clarification:**
- Store `last_triggered` **in UTC**.
- Convert `time` field (HH:MM) to UTC **for comparison only** using the user's `timezone` field.
- Use a 23-hour window **in UTC** to prevent double-firing.

**Add to Requirement 2:**
> *"2.7 THE Checkpoint_Runner SHALL store `last_triggered` in UTC. WHEN comparing `last_triggered` to current time, THE Checkpoint_Runner SHALL apply the user's configured `timezone` to determine if the checkpoint is due."*

---

#### Gap 3: WAR_MODE Auto-Expiry Mechanism (Undefined)

**Location:** Requirement 7.4

**Issue:** *"WHEN `war_mode_expires` is reached, THE Bot SHALL automatically set `current_mode` back to `NORMAL`."*

How does the system **know** the time has expired? There is no cron job defined for this. The Checkpoint_Runner only checks **checkpoints**, not user state expiry.

**Required Clarification:**
- **Option A:** Add a second cron job that runs every 5 minutes: `expire_war_modes()`.
- **Option B:** Check `war_mode_expires` during **every** Checkpoint_Runner execution (already runs every minute). If `war_mode_expires < now()`, flip mode to `NORMAL` **before** processing checkpoints.

**Recommendation:** Option B. Efficient. No new cron job.

**Add to Requirement 7:**
> *"7.6 WHEN the Checkpoint_Runner executes, it SHALL evaluate `war_mode_expires` for all users in `WAR_MODE`. IF `war_mode_expires` is less than the current UTC time, THE Checkpoint_Runner SHALL set `current_mode` to `NORMAL` and clear `war_mode_expires` before processing any checkpoints for that user."*

---

#### Gap 4: Image Evaluation Confidence Threshold (Too Rigid)

**Location:** Requirement 10.6

**Issue:** *"IF the Agent cannot determine a verdict with confidence above 0.5, THEN THE Agent SHALL assign FAILED."*

For a photo of food, this is fine. For a photo of a **gym locker room** at 6:00 AM, a blurry image might be genuine but low confidence. You risk **false positives** (punishing success).

**Required Clarification:**
- For `IMAGE_DEMAND` checkpoints with `response_validation.expected_type` = `GYM_PROOF` or `FOOD`, **lower the threshold to 0.3** and request clarification instead of auto-failing.
- Auto-fail **only** if the image is clearly wrong (e.g., a screenshot, a black screen).

**Modify Requirement 10.6:**
> *"10.6 IF the Agent cannot determine a verdict with confidence above 0.5 for TEXT responses, or above 0.3 for IMAGE responses, THEN THE Agent SHALL request clarification from the user before assigning FAILED."*

---

#### Gap 5: `EOD_Report` vs. `Morning_Todo` Race Condition

**Location:** Requirement 17, Requirement 18

**Issue:** Requirement 17.4 stores `next_day_plan` with `date` = tomorrow. Requirement 18.3 reads `next_day_plan` for the **previous night**. What happens if the EOD report **fails to generate**? The Morning Todo has no plan to read.

**Required Clarification:**
- Requirement 18.4 handles this with fallback to default schedule. This is correct.
- **But:** The fallback must also **log a warning** and **increment a metric** (`morning_todo_fallback_count`) so Chanakya knows EOD is failing.

**Add to Requirement 18.4:**
> *"WHEN the fallback to default schedule occurs, THE Bot SHALL increment `morning_todo_fallback_count` in the user's MongoDB document and include the fallback reason in the `Morning_Todo` message (e.g., 'Plan missing. Using default schedule. Fix EOD.')"*

---

#### Gap 6: `checkin_window` Timezone Handling

**Location:** Requirement 19.2

**Issue:** *"WHEN the current time falls within the user's configured active-hours window..."*

Same timezone ambiguity as Gap 2. The server runs UTC. The user's window is in IST.

**Required Clarification:**
- Store `checkin_window_start` and `checkin_window_end` as **HH:MM strings**.
- The Checkpoint_Runner must convert current UTC time to user's `timezone` **before** checking if it falls in the window.

**Add to Requirement 19:**
> *"19.9 THE Checkpoint_Runner SHALL convert the current UTC time to the user's configured `timezone` before evaluating whether the current time falls within `checkin_window_start` and `checkin_window_end`."*

---

### Part 3: Edge Cases Not Addressed

| Edge Case | Impact | Required Addition |
| :--- | :--- | :--- |
| **User sends photo without caption** | Agent has no text context. | Req 3.7: Add *"IF photo has no caption, THE Bot SHALL treat the caption as an empty string and rely solely on image analysis."* |
| **Twilio call answered by voicemail** | Voicemail records "Hello?" Agent thinks user spoke. | Req 13: Add *"13.6 THE Bot SHALL treat calls under 10 seconds as VOICEMAIL and mark the interaction as FAILED."* |
| **User deletes Telegram message** | Bot cannot see edit/delete events easily. | Req 3: Add *"3.8 THE Bot SHALL ignore message edits and deletions; only the initial message is evaluated."* |
| **MongoDB Atlas connection drops** | Checkpoint_Runner crashes. | Req 2: Add *"2.8 THE Checkpoint_Runner SHALL implement exponential backoff retry logic for MongoDB connection failures."* |
| **ElevenLabs credit exhausted** | Voice calls fail mid-script. | Req 14: Add *"14.5 THE Bot SHALL monitor ElevenLabs credit balance via API and alert the user at 1000 credits remaining."* |

---

### Part 4: MongoDB Schema Alignment Issues

Your requirements describe fields that are not present in the schema I provided earlier. We must align.

| Requirement Field | Collection | Schema Status | Action |
| :--- | :--- | :--- | :--- |
| `streak_count` | `users` | **Missing** | Add to `users` schema. |
| `failure_count_this_week` | `users` | **Missing** | Add to `users` schema (computed field). |
| `longest_streak` | `users` | **Missing** | Add to `users` schema. |
| `morning_todo_time` | `users` | **Missing** | Add to `users` schema. |
| `morning_todo_fallback_count` | `users` | **Missing** | Add to `users` schema. |
| `checkin_window_start` / `_end` | `users` | **Missing** | Add to `users` schema. |
| `checkin_min_per_day` / `_max` | `users` | **Missing** | Add to `users` schema. |
| `current_activity` | `users` | **Missing** | Add to `users` schema. |
| `activity_slot_updated_at` | `users` | **Missing** | Add to `users` schema. |
| `next_day_plan` (object) | `users` | **Missing** | Add to `users` schema with `{date, plan_text, confirmed}`. |
| `timezone` | `users` | **Missing** | Add to `users` schema (default `Asia/Kolkata`). |
| `prompt_templates` | **New Collection** | **Missing Entirely** | Requirement 20.10 requires this collection. |
| `recurring_failure_patterns` | `users` or separate | **Missing** | Requirement 20.4 references pre-computed patterns. |

**Required Action:** I will provide an **Updated MongoDB Schema** in the next message that includes all these fields.

---

### Part 5: Revised / Additional Requirements

Based on the gaps above, here are the **mandatory additions**:

#### Requirement 22: MongoDB Schema Completeness

**User Story:** As a developer, I want all fields referenced in the requirements to exist in the MongoDB schema, so that no runtime `KeyError` occurs.

**Acceptance Criteria:**
1. THE `users` collection SHALL include the following fields with appropriate types and defaults: `streak_count` (int, default 0), `longest_streak` (int, default 0), `failure_count_this_week` (int, default 0), `morning_todo_time` (string, null), `morning_todo_fallback_count` (int, default 0), `checkin_window_start` (string, default "09:00"), `checkin_window_end` (string, default "21:00"), `checkin_min_per_day` (int, default 2), `checkin_max_per_day` (int, default 4), `current_activity` (string, default "FREE_TIME"), `activity_slot_updated_at` (datetime, null), `next_day_plan` (object, default `{}`), `timezone` (string, default "Asia/Kolkata").
2. THE `prompt_templates` collection SHALL exist with documents containing `activity_slot`, `interaction_type`, `tone`, `template_text`, and `version`.
3. THE `users` collection SHALL include `recurring_failure_patterns` (array of objects) for pre-computed pattern storage.

---

#### Requirement 23: Prompt Template Management

**User Story:** As a user, I want all prompt text stored in MongoDB and selectable by Activity_Slot and tone, so that the system's voice can be tuned without code deployment.

**Acceptance Criteria:**
1. THE `prompt_templates` collection SHALL store documents with `activity_slot` (ENUM: `OFFICE_WORK`, `LEETCODE`, `GYM`, `FREE_TIME`, `SLEEP`), `interaction_type` (ENUM: `CHECKPOINT`, `CHECK_IN`, `EOD`, `ESCALATION`, `MENTOR_TALK`, `COMMAND_RESPONSE`), `tone` (ENUM: `HARSH`, `MENTOR`, `NEUTRAL`, `CELEBRATORY`), `template_text` (string with `{variable}` placeholders), and `version` (int).
2. WHEN the Agent requires a prompt, it SHALL query `prompt_templates` with `activity_slot`, `interaction_type`, and the computed `tone` (based on user's recent performance).
3. IF no matching template exists, THE Agent SHALL fall back to `activity_slot: FREE_TIME` with the same `interaction_type` and `tone`.
4. IF still no match, THE Agent SHALL fall back to a generic template stored in environment variables.

---

#### Requirement 24: Timezone Consistency Across All Time-Based Operations

**User Story:** As a user, I want all time-based operations to respect my configured timezone, so that checkpoints fire at the correct local time regardless of server location.

**Acceptance Criteria:**
1. THE Checkpoint_Runner SHALL convert all `HH:MM` times in `checkpoints` to the user's configured `timezone` before comparing to current server time.
2. ALL `last_triggered` timestamps SHALL be stored in UTC.
3. ALL `created_at` and `updated_at` timestamps SHALL be stored in UTC.
4. WHEN computing `failure_count_this_week`, THE Bot SHALL use the user's `timezone` to determine the start of the local week (Monday 00:00).
5. WHEN generating `User_State_Snapshot` with `date` field, THE Bot SHALL use the user's `timezone` to determine the calendar date boundary.

---

### Summary: Document Health Check

| Category | Score | Notes |
| :--- | :--- | :--- |
| **Completeness** | 85% | Missing schema fields and edge cases. |
| **Clarity** | 90% | Glossary is excellent. Few ambiguous terms. |
| **Testability** | 95% | Every requirement has measurable ACs. |
| **Scalability** | 100% | Multi-tenant, context-tiered, prompt-driven. |
| **Chanakya Alignment** | 100% | Harsh, logical, self-correcting. |

---