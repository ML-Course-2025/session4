# Activity 1

### Objective:

- Use the following prompt to generate a detailed specification for **Phase 1** of the project, as well as general specifications for **Phase 2** and **Phase 3**.
- Review the provided **sample specification** [here](./specs-sample.md). This will give you an example of how the specifications should look.
<!-- - Generate [Demo](./demo.ipynb)  -->

----

```markdown
We are a group of four students learning a beginner-level course on machine learning. Most members of our group are beginners in both ML and programming.

We want to create a collaborative project where different members contribute to different tasks. Our goal is to build an AI-powered application with a user interface. This application should receive data from users, perform some basic filtering/validation on the input, and then send it to a large language model (LLM) for processing (e.g., to generate text, summarize, or answer questions based on the input).

We are still deciding on the application’s specific focus, considering three general areas:
1) Healthcare
2) Commerce/Business
3) Security

Our project will be developed in **three phases**:
- **Phase 1 (1 week):** Prototyping – Creating a mock UI demo using Gradio that receives input and returns hardcoded mock data relevant to our chosen application idea.
- **Phase 2 (1 week):** Implementing LLM-based functionality – Replacing the mock logic with actual AI processing. For the AI backend, we plan to select, load, and utilize a small language model from Hugging Face, ensuring it runs effectively within Google Colab's Free Tier limits. If this is not feasible, we will integrate an external LLM API. The implementation will be primarily developed with LLM assistance ("vibe coding").
- **Phase 3 (3 days):** Documentation, presentation preparation, and finalizing the product.

For development, we will use:
- **Gradio** for creating the user interface
- **Google Colab Free Tier** (utilizing the free GPU if needed for the LLM interaction later)
- A **"vibe coding"** approach, where an LLM assists heavily in generating the code.

Please **ask us one question at a time** to help us refine:
1.  **Full specifications for Phase 1** (the mock GUI demo). These should be detailed enough to build from.
2.  **General specifications for Phase 2** (LLM integration).
3.  **General specifications for Phase 3** (final steps).

Please ensure the proposed specifications outline a project scope that is realistically achievable for beginners within the given timeframe and tools. The specifications should also ideally suggest potential ways the work could be divided among the four team members for each phase. Start by helping us narrow down a specific application idea within one of the domains (Healthcare, Commerce, Security).
```


