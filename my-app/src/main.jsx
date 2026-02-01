import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)

// Custom cursor implementation
const createCursor = () => {
  // Create cursor elements
  const cursorDot = document.createElement('div');
  cursorDot.className = 'cursor-dot';
  
  const cursorOutline = document.createElement('div');
  cursorOutline.className = 'cursor-outline';
  
  document.body.appendChild(cursorDot);
  document.body.appendChild(cursorOutline);
  
  let mouseX = 0;
  let mouseY = 0;
  let cursorX = 0;
  let cursorY = 0;
  let outlineX = 0;
  let outlineY = 0;
  
  // Update mouse position
  document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  });
  
  // Animate cursor
  const animate = () => {
    // Smooth follow with different speeds (increased for lighter feel)
    cursorX += (mouseX - cursorX) * 0.5;
    cursorY += (mouseY - cursorY) * 0.5;
    
    outlineX += (mouseX - outlineX) * 0.25;
    outlineY += (mouseY - outlineY) * 0.25;
    
    cursorDot.style.left = `${cursorX}px`;
    cursorDot.style.top = `${cursorY}px`;
    
    cursorOutline.style.left = `${outlineX}px`;
    cursorOutline.style.top = `${outlineY}px`;
    
    requestAnimationFrame(animate);
  };
  
  animate();
  
  // Add hover effects
  const addHoverListeners = () => {
    const hoverElements = document.querySelectorAll('a, button, input, textarea, select, [role="button"], .module-card, .roi-button, .toggle-button');
    
    hoverElements.forEach(el => {
      el.addEventListener('mouseenter', () => {
        cursorDot.classList.add('hover');
        cursorOutline.classList.add('hover');
      });
      
      el.addEventListener('mouseleave', () => {
        cursorDot.classList.remove('hover');
        cursorOutline.classList.remove('hover');
      });
    });
  };
  
  // Initial setup
  addHoverListeners();
  
  // Re-add listeners when DOM changes (for dynamic content)
  const observer = new MutationObserver(addHoverListeners);
  observer.observe(document.body, { childList: true, subtree: true });
};

// Initialize cursor after DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', createCursor);
} else {
  createCursor();
}

// to run : npm run dev