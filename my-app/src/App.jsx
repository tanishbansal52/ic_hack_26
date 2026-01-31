import React from 'react';
import ModuleCard from './ModuleCard';

// Hardcoded list from your DB for now
const modules = [
  { id: 1, code: 'DOC601', name: 'Machine Learning' },
  { id: 2, code: 'DOC602', name: 'Computer Graphics' },
  { id: 3, code: 'DOC404', name: 'Professional Skills' }
];

function App() {
  return (
    <div style={{ padding: '40px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Module Selection Helper</h1>
      <p>Select a module to see if attendance actually matters.</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '20px' }}>
        {modules.map(mod => (
          <ModuleCard key={mod.id} module={mod} />
        ))}
      </div>
    </div>
  );
}

export default App;