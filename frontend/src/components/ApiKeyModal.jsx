import { useState } from 'react';
import { KeyRound } from 'lucide-react';

export default function ApiKeyModal({ onApiKeySubmit }) {
  const [apiKey, setApiKey] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (apiKey.trim()) {
      onApiKeySubmit(apiKey);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 backdrop-blur">
      <div className="bg-white rounded-lg shadow-2xl p-8 w-full max-w-md">
        <div className="flex flex-col items-center text-center">
          <div className="bg-blue-100 p-3 rounded-full mb-4">
            <KeyRound className="text-blue-600" size={28} />
          </div>
          <h2 className="text-2xl font-bold mb-2">Enter Your Google API Key</h2>
          <p className="text-slate-500 mb-6">Your API key is stored locally in your browser and is never sent anywhere except to Google's API.</p>
          <form onSubmit={handleSubmit} className="w-full">
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              className="w-full p-3 border border-slate-300 rounded-md text-center focus:ring-2 focus:ring-blue-500 focus:outline-none"
              placeholder="••••••••••••••••••••••••••••••"
            />
            <button
              type="submit"
              className="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-md hover:bg-blue-700 mt-4 transition-colors"
            >
              Save API Key
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}