import { useState } from "react";
import axios from "axios";
import { FaStar } from "react-icons/fa";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm"; // For bullet points, bold, tables, etc.


function App() {
  const [question, setQuestion] = useState("");
  const [solution, setSolution] = useState("");
  const [loading, setLoading] = useState(false);
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(null);
  const [comments, setComments] = useState("");
  const [improved, setImproved] = useState("");

  const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

  const askQuestion = async () => {
    if (!question.trim()) {
      alert("Please enter a math question first.");
      return;
    }
    setLoading(true);
    setSolution("");
    try {
      const res = await axios.post(`${API_BASE}/ask`, { question });
      setSolution(res.data.solution);
    } catch (err) {
      alert(err.response?.data?.detail || "An error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async () => {
    if (!rating) {
      alert("Please give a star rating before submitting feedback.");
      return;
    }
    try {
      await axios.post(`${API_BASE}/feedback`, {
        question,
        response: solution,
        rating,
        comments,
        improved_response: improved,
      });
      alert("✅ Feedback saved successfully!");
      setComments("");
      setImproved("");
      setRating(0);
    } catch (err) {
      alert("Error saving feedback.");
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 flex flex-col items-center p-6">
      <div className="w-full max-w-3xl bg-white shadow-md rounded-2xl p-8 mt-10 border border-gray-100">
        <h1 className="text-3xl font-bold mb-4 text-center text-blue-700">
          🎓 Math Professor AI
        </h1>
        <p className="text-center text-gray-600 mb-6">
          Ask your mathematics question and get a step-by-step solution.
        </p>

        {/* Input Box */}
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Example: Solve for x if 2x + 5 = 15"
          className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent mb-4 resize-none"
          rows="3"
        />

        <button
          onClick={askQuestion}
          disabled={loading}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg shadow-md transition duration-150 disabled:opacity-50"
        >
          {loading ? (
            <span className="flex justify-center items-center space-x-2">
              <span className="animate-spin border-2 border-t-transparent border-white rounded-full w-5 h-5"></span>
              <span>Solving...</span>
            </span>
          ) : (
            "🚀 Get Solution"
          )}
        </button>

        {/* Solution Section */}
        {solution && (
          <div className="mt-8 border-t border-gray-200 pt-6">
            <h2 className="text-xl font-semibold mb-3 text-green-700 flex items-center">
              📘 Solution
            </h2>
            <pre className="bg-gray-50 p-4 rounded-lg border border-gray-200
                            prose prose-blue max-w-none overflow-x-auto break-words
                            whitespace-pre-wrap sm:text-base text-sm leading-relaxed">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {solution}
              </ReactMarkdown>
            </pre>

            {/* Feedback Section */}
            <div className="mt-6">
              <h3 className="font-semibold text-lg mb-2 text-gray-800">
                💬 Rate and Provide Feedback
              </h3>

              {/* Star Rating */}
              <div className="flex space-x-1 mb-3">
                {[1, 2, 3, 4, 5].map((star) => (
                  <FaStar
                    key={star}
                    size={26}
                    className={`cursor-pointer transition-colors duration-200 ${
                      (hover || rating) >= star
                        ? "text-yellow-400"
                        : "text-gray-300"
                    }`}
                    onClick={() => setRating(star)}
                    onMouseEnter={() => setHover(star)}
                    onMouseLeave={() => setHover(null)}
                  />
                ))}
              </div>

              {/* Comment Box */}
              <textarea
                placeholder="Comments or suggestions..."
                className="w-full p-3 border border-gray-300 rounded-lg mb-3 focus:ring-2 focus:ring-blue-400 resize-none"
                value={comments}
                onChange={(e) => setComments(e.target.value)}
                rows="3"
              />

              {/* Improved Solution */}
              <textarea
                placeholder="Your improved solution (optional)..."
                className="w-full p-3 border border-gray-300 rounded-lg mb-3 focus:ring-2 focus:ring-green-400 resize-none"
                value={improved}
                onChange={(e) => setImproved(e.target.value)}
                rows="3"
              />

              <button
                onClick={submitFeedback}
                className="bg-green-600 hover:bg-green-700 text-white font-semibold px-4 py-2 rounded-lg shadow-md transition duration-150"
              >
                Submit Feedback
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <p className="text-gray-500 text-sm mt-10">
        Built with ❤️ using FastAPI + React + Tailwind
      </p>
    </div>
  );
}

export default App;
