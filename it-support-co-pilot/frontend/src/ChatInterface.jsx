import React, { useState } from 'react';
import axios from 'axios';
import './index.css' 

//URL of your FastAPI backend
const API_URL = 'http://localhost:8000/resolve';

//initial state for the result display
const initialResult = {
    final_response: "Enter your IT issue and User ID to get a resolution.",
    agent_status: "Awaiting Query",
    suggested_jira_ticket: "None",
    raw_classification: {
        category: "N/A",
        asset_id: "N/A"
    }
};

function ChatInterface() {
    const [query, setQuery] = useState("");
    const [userId, setUserId] = useState("pungkj"); //default user for quick testing
    const [result, setResult] = useState(initialResult);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);
        setError(null);
        setResult(initialResult); //reset previous results

        //1. Prepare the structured request payload (matching the QueryRequest Pydantic schema)
        const requestPayload = {
            user_query: query,
            user_id: userId,
        };

        try {
            //2. Make the POST request to your FastAPI endpoint
            const response = await axios.post(API_URL, requestPayload, {
                headers: { 'Content-Type': 'application/json' }
            });

            //3. Update state with the structured response (matching the Resolution Pydantic schema)
            setResult(response.data);
        } catch (err) {
            console.error("API Error:", err);
            setError(`Failed to connect or process query. Details: ${err.response?.data?.detail || err.message}`);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-100 p-8">
            <h1 className="text-3xl font-bold mb-8 text-center text-indigo-700">
                AI IT Support Co-pilot 🤖
            </h1>
            <div className="max-w-4xl mx-auto bg-white p-6 rounded-xl shadow-2xl">
                
                {/*input form*/}
                <form onSubmit={handleSubmit} className="mb-6 space-y-4">
                    <input
                        type="text"
                        value={userId}
                        onChange={(e) => setUserId(e.target.value)}
                        placeholder="Your User ID (e.g., pungkj)"
                        required
                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                    />
                    <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Describe your IT issue (e.g., My laptop LAP-4567 cannot connect to the WiFi on the 3rd floor.)"
                        required
                        rows="4"
                        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-indigo-500 focus:border-indigo-500"
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !query || !userId}
                        className={`w-full py-3 rounded-lg text-white font-semibold transition duration-300 ${
                            isLoading ? 'bg-indigo-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'
                        }`}
                    >
                        {isLoading ? 'Processing by Multi-Agent Workflow...' : 'Resolve Issue'}
                    </button>
                </form>

                {error && (
                    <div className="p-4 bg-red-100 text-red-700 border border-red-300 rounded-lg mb-4">
                        <p className="font-bold">Error:</p>
                        <p>{error}</p>
                    </div>
                )}
                
                {/*output display*/}
                <div className="space-y-6">
                    <h2 className="text-2xl font-semibold border-b pb-2 text-indigo-700">Resolution Output</h2>
                    
                    {/* Final Response (Resolver Agent's Output) */}
                    <div className="bg-green-50 p-4 rounded-lg border-l-4 border-green-600 shadow-md">
                        <p className="font-bold text-green-800 mb-1">Final Resolution:</p>
                        <p className="whitespace-pre-wrap text-gray-800">{result.final_response}</p>
                    </div>

                    {/* Agent Status and Metrics (The complex showcase) */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <ResultBox title="Agent Status" value={result.agent_status} color="bg-blue-100 border-blue-600 text-blue-800" />
                        <ResultBox title="Mock JIRA/Automation" value={result.suggested_jira_ticket} color="bg-yellow-100 border-yellow-600 text-yellow-800" />
                        <ResultBox title="Classifier Category" value={result.raw_classification.category} color="bg-purple-100 border-purple-600 text-purple-800" />
                        <ResultBox title="Identified Asset ID" value={result.raw_classification.asset_id} color="bg-purple-100 border-purple-600 text-purple-800" />
                    </div>
                </div>
            </div>
        </div>
    );
}

const ResultBox = ({ title, value, color }) => (
    <div className={`${color} p-4 rounded-lg border-l-4 shadow-sm`}>
        <p className="font-medium text-sm">{title}</p>
        <p className="text-xl font-bold mt-1 bg-red">{value}</p>
    </div>
);

export default ChatInterface;