import React, { useState } from "react";

function App() {
  const [url, setUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [transcript, setTranscript] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setAnswer("");
    setTranscript("");
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:8000/youtube-qa/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url, question }),
      });

      const data = await response.json();

      if (data.error) {
        setError(data.error);
      } else {
        setAnswer(data.answer);
        setTranscript(data.transcript);
      }
    } catch (err) {
      setError("Something went wrong while fetching the answer.");
    }
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">YouTube Q&A App</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <input
          type="text"
          placeholder="Enter YouTube URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          className="w-full p-2 border rounded"
          required
        />
        <input
          type="text"
          placeholder="Enter your question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          className="w-full p-2 border rounded"
          required
        />
        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          Submit
        </button>
      </form>

      {error && <p className="text-red-600 mt-4">{error}</p>}
      {answer && (
        <div className="mt-6">
          <h2 className="font-semibold">Answer:</h2>
          <p>{answer}</p>
        </div>
      )}
      {transcript && (
        <div className="mt-6">
          <h2 className="font-semibold">Transcript:</h2>
          <pre className="whitespace-pre-wrap">{transcript}</pre>
        </div>
      )}
    </div>
  );
}

export default App;

