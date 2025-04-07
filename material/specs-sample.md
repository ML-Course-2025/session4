# Sample Project Specifications

This spec document reflects the decisions made during the conversation with the LLM, focusing on the **Mental Health Support App** with dynamic dropdowns.

---

## AI-Powered Mental Health Support Assistant (Prototype)**

This document outlines the specifications for our group project, derived from an interactive session with an AI assistant. We are a team of four beginners building an application using Gradio, Google Colab, and leveraging LLM assistance ("vibe coding"). Our goal is to create a supportive tool while emphasizing it is **not a replacement for professional medical advice or crisis intervention.**

**Chosen Application Focus:** Healthcare -> Mental Health Support

### **Phase 1: Mock UI Demo (1 Week) - Detailed Specifications**

**Goal:** Create a functional Gradio user interface prototype that simulates the user interaction flow for the Mental Health Support app. This prototype will accept user selections via multiple dropdowns and display *hardcoded* mock data relevant to the selections, without actual AI processing.

**1. User Interface (Gradio):** The interface will consist of the following components, arranged vertically:

    *   **Input Component 1 (Dropdown):** Labeled "Select Main Category".
        *   Default Option: "-- Select a mental health topic --" (Mandatory selection required to proceed).
        *   Options: `["Stress & Anxiety", "Depression", "Sleep Issues", "Emotional Well-being", "General Mental Health Support"]`
    *   **Input Component 2 (Dropdown):** Labeled "Select Subcategory".
        *   Default Option: "-- Select a subtopic --"
        *   Options: Initially, this might show a combined list or be disabled until the Main Category is selected. **In Phase 1, the dynamic update logic is NOT required**, but the dropdown should exist. Mock logic will *pretend* to use this selection. Example options (static list for Phase 1, to be made dynamic later): `["Coping Techniques", "Breathing Exercises", "Mindfulness Tips", "Motivational Support", "Journaling Prompts", "Professional Help Resources", "Sleep Hygiene Tips", "Relaxation Techniques", "Guided Sleep Meditation", "Self-care Activities", "Positive Affirmations", "Relationship Advice"]`
    *   **Input Component 3 (Dropdown):** Labeled "Select Severity Level".
        *   Default Option: "-- Select severity --"
        *   Options: `["Mild", "Moderate", "Severe", "Crisis"]`
    *   **Input Component 4 (Dropdown):** Labeled "Select Age Group".
        *   Default Option: "-- Select age group --"
        *   Options: `["Teen (13-19 years)", "Young Adult (20-30 years)", "Adult (31-50 years)", "Senior (50+ years)"]`
    *   **Button:** Labeled "Get Support Tips".
    *   **Output Component 1 (Text Area):** Labeled "Your Selections".
        *   Purpose: To display the specific options chosen by the user from all four dropdowns after clicking the button.
    *   **Output Component 2 (Text Area):** Labeled "Suggested Support (Mock Data)".
        *   Purpose: To display a static, pre-written response simulating helpful advice or resources.

**2. Core Functionality (Mock Logic):**
    *   When the user has made selections in all dropdowns and clicks the "Get Support Tips" button:
        *   The application reads the selected values from all four dropdowns.
        *   The application displays the selected values in the "Your Selections" output area.
        *   The application displays a *fixed, hardcoded* support message in the "Suggested Support (Mock Data)" output area.
        *   **Crucially, in Phase 1, the hardcoded response will be the same (or one of a few pre-written options selected randomly/simply) regardless of the specific dropdown combination chosen.** The goal is to show the UI structure and flow.
    *   **Example Hardcoded Output:**
        ```
        Based on your selections (Main Category: [User's Choice], Subcategory: [User's Choice], Severity: [User's Choice], Age Group: [User's Choice]):

        Here are some general tips that might be helpful:
        *   Practice deep breathing exercises for 5 minutes daily.
        *   Consider journaling your thoughts and feelings.
        *   Ensure you are getting adequate sleep and nutrition.
        *   Reach out to a trusted friend or family member.

        **Disclaimer:** This tool provides general suggestions and is not a substitute for professional medical advice, diagnosis, or treatment. If you selected 'Severe' or 'Crisis', or feel you need immediate help, please contact a crisis hotline or mental health professional immediately. [Include example hotline number/link].

        (Note: This is a static example for the Phase 1 demo.)
        ```

**3. Technology:**
    *   Language: Python 3
    *   UI Library: Gradio
    *   Development Environment: Google Colab (Free Tier)

**4. Deliverable:**
    *   A runnable Google Colab notebook (`.ipynb`) containing the Python code that launches the functional Gradio mock UI demo with the four dropdowns, button, and output areas.

**5. Suggested Task Division (Example for 4 members):**
    *   **Member 1 (UI Setup & Dropdowns 1 & 2):** Set up Colab, import Gradio, create the interface structure. Implement the "Main Category" and "Subcategory" dropdowns.
    *   **Member 2 (Dropdowns 3 & 4 & Button):** Implement the "Severity Level" and "Age Group" dropdowns and the "Get Support Tips" button. Define the function structure called by the button.
    *   **Member 3 (Input Display & Mock Response):** Implement the logic within the button's function to read all dropdown values and display them in the "Your Selections" output. Write the hardcoded mock response text.
    *   **Member 4 (Output Display & Integration):** Implement the "Suggested Support" output area. Integrate the display of the mock response. Combine all parts, test the flow, ensure the notebook runs.

---

### **Phase 2: LLM-Based Functionality (1 Week) - General Specifications**

**Goal:** Enhance the prototype by replacing the hardcoded mock response with dynamically generated suggestions using an AI model (local Hugging Face model preferred, API as fallback) and implement the dynamic update for the Subcategory dropdown.

**1. AI Integration:**
    *   Implement AI logic using either:
        *   **Primary Path:** A small, suitable text-generation model from Hugging Face runnable on Colab Free Tier.
        *   **Fallback Path:** An external LLM API (e.g., Gemini free tier).
    *   Handle API keys securely if using the fallback.

**2. Dynamic UI Logic:**
    *   Implement the Python/Gradio logic to make the "Select Subcategory" dropdown (Input Component 2) dynamically update its options based on the selection made in the "Select Main Category" dropdown (Input Component 1), using the mappings defined in the Q&A.

**3. Input Processing (Prompt Engineering):**
    *   Construct a prompt for the LLM based on the user's selections from *all four* dropdowns (Main Category, Subcategory, Severity, Age Group).
    *   Example Prompt Structure: `"Act as a supportive mental health assistant (not a therapist). Based on the user's selection of Main Category: [Category], Subcategory: [Subcategory], Severity: [Severity], and Age Group: [Age Group], provide brief, empathetic, actionable wellness tips or resources. If severity is 'Severe' or 'Crisis', strongly emphasize seeking professional help and provide crisis resource information instead of general tips. Keep the tone supportive and general."`

**4. Output Handling & Safety:**
    *   Call the model/API with the prompt.
    *   Display the AI-generated response in the output area.
    *   **Implement specific logic:** If "Severity" is "Severe" or "Crisis", bypass standard AI generation and display predefined crisis resources and disclaimers urging professional help.
    *   Include disclaimers in all responses about not being professional advice.

**5. Technology:**
    *   Python, Gradio, Google Colab
    *   Hugging Face libraries (`transformers`, etc.) OR API client libraries.

**6. Suggested Task Division (General):**
    *   **AI Research & Backend Setup:** Focus on getting the chosen model/API working within Colab.
    *   **Dynamic UI Implementation:** Implement the logic for the dependent dropdowns in Gradio.
    *   **Prompt Engineering & Integration:** Develop prompt creation logic and integrate the AI call.
    *   **Output Handling & Safety:** Implement the display of AI responses, including the critical safety logic for high severity levels.

---

### **Phase 3: Finalization (3 Days) - General Specifications**

**Goal:** Polish the application, ensure safety disclaimers are prominent, document thoroughly, and prepare for the final presentation.

**1. Code Refinement & Safety Check:**
    *   Clean up code, add comments.
    *   Rigorously test the safety logic for "Severe" and "Crisis" severity levels. Ensure appropriate resources and disclaimers are always shown in these cases.
    *   Verify general disclaimers are present in all outputs.

**2. Documentation (`README.md`):**
    *   Create `README.md` covering: Project Goal (Mental Health Support Tool), Team, How to Run, Features (including dynamic dropdowns, AI integration choice), Architecture, Safety Considerations/Disclaimer Emphasis, Challenges, Future Ideas.

**3. Presentation Preparation:**
    *   Create slides covering Introduction, Demo (showcasing dynamic UI and AI response, including safety case), Technical Details, Safety Measures, Learnings.
    *   Practice the demo flow.

**4. Final Testing:**
    *   Test various combinations of inputs, focusing on edge cases and the dynamic UI behavior.

**5. Deliverables:**
    *   Final Colab notebook (`.ipynb`).
    *   `README.md`.
    *   Presentation slides.

**6. Suggested Task Division (General):**
    *   **Code & Safety Lead:** Final code polish, rigorous testing of safety features.
    *   **Documentation Lead:** Draft `README.md`, ensuring safety aspects are highlighted.
    *   **Presentation Lead:** Create slides, coordinate presentation structure.
    *   **Testing & Coordination Lead:** Final testing, bug checks, content gathering.
    *   *(All members contribute, review safety aspects, and practice.)*

