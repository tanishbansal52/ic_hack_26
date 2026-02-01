## Inspiration
We noticed how difficult it is for educators to gauge student engagement in large lecture halls. Traditional methods rely on gut feeling or manual observation, which doesn't scale. We wanted to create a tool that could automatically track attention levels during lectures, helping professors understand which topics resonate and which lose students' focus.

## What it does
Modulator is an AI-powered lecture analytics platform that combines computer vision and intelligent agents to measure student engagement in real-time. Using webcam feeds, it detects people in the frame, identifies individual students across sessions, and tracks attention metrics like eye gaze and head orientation. The system correlates this engagement data with lecture content to identify which modules are most and least effective, presenting insights through an interactive web dashboard.

## How we built it
We built Modulator with three main components: a computer vision pipeline using YOLO for person detection, InsightFace for facial recognition, and MediaPipe for attention tracking; a Flask backend with SQLite for storing student identities and engagement metrics; and a React frontend for visualizing the data. We also integrated an AI agent using ADK to provide intelligent insights about module effectiveness based on the collected data.
## Challenges we ran into
The initial vision model we used was insufficient for scoring attentiveness. As such, we conducted a review of a multitude of other models, evaluating the alternatives based on accuracy, robustness and suitability for real-time facial recognition.

To uphold ethical standards, the personal features gleaned from the vision model are stored using abstract vectors, ensuring face captures are anonymised

In order for the AI to reason well with the module data give, we provided it with structure contextual grounding. This exemplified its performance when dealing with the data when compared to a vanilla AI agent.

## Accomplishments that we're proud of
Data-driven study optimisation: We have developed a system that enables students to make informed decisions about their attendance, helping them prioritise lectures that are statistically linked to higher grades.
Privacy-first architecture: We have engineered a secure facial recognition pipeline that anonymises user data instantly, ensuring that no raw images are stored — only mathematical embeddings.
Intelligent advisory assistant: We have integrated an AI-powered chatbot that translates complex statistical data into personalised, actionable academic advice for students.
## What we learned
We mastered the complexity of orchestrating a multi-stage computer vision pipeline in real-time. Integrating YOLO, InsightFace, and MediaPipe required precise synchronization to ensure accurate detection without lagging the video feed. We also deepened our understanding of vector database management, specifically optimizing SQLite for high-speed embedding retrieval. Beyond the code, we learned valuable lessons in release engineering—navigating strict dependency conflicts and pinning stable Python versions to keep our ML stack robust.

## What's next for Modulator
We hope to pitch this to every department in every university so this can hopefully become a tool useful for both lecturers and students alike! We’d like to expand into incorporating students feedback on the app as well, considering improvements based on real experiences and integrating suggestions that enhance both teaching effectiveness and student outcomes.