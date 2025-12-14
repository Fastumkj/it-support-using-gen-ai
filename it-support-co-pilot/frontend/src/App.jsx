import ChatInterface from './ChatInterface';
import './index.css' 

function App() {
  return (
    <div>
      <div className="App">
        <ChatInterface />
      </div>
      <div className="bg-red-500 min-h-screen">
        <ChatInterface />
      </div>
    </div>
  );
}

export default App;