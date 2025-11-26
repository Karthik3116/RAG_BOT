import { useState } from 'react';
import ChatPage from './pages/ChatPage';
import ApiKeyModal from '././components/ApiKeyModal';
import { useLocalStorage } from '././hooks/useLocalStorage';

export default function App() {
  const [apiKey, setApiKey] = useLocalStorage('googleApiKey', null);

  const handleApiKeySubmit = (newApiKey) => {
    setApiKey(newApiKey);
  };

  if (!apiKey) {
    return <ApiKeyModal onApiKeySubmit={handleApiKeySubmit} />;
  }

  return <ChatPage apiKey={apiKey} />;
}