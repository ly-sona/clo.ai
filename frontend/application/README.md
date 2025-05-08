# clo.ai Frontend

<p align="center">
  <img src="public/genetic-algorithm-svgrepo-com.svg" alt="clo.ai logo" width="120" height="120">
</p>

Modern React frontend for the VLSI layout optimization system.

## Features

- Interactive circuit visualization using ReactFlow
- Drag-and-drop file uploads for circuit optimization
- Real-time optimization progress tracking
- Side-by-side comparison of original vs. optimized circuits
- Downloadable optimized circuit files

## Tech Stack

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: TailwindCSS
- **UI Components**: Shadcn/UI (accessible components)
- **Animations**: Framer Motion
- **Circuit Visualization**: ReactFlow
- **Network Requests**: Fetch API

## Development

### Prerequisites

- Node.js 18+
- npm 9+

### Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open your browser to http://localhost:5173

### Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

### Linting

```bash
npm run lint
```

## Project Structure

```
src/
├── api/           # API client functions
├── assets/        # Static assets
├── components/    # React components
│   ├── ui/        # Shadcn/UI components
│   └── ...        # Custom components
├── pages/         # Page components
├── styles/        # Global styles
├── App.tsx        # Root component
└── main.tsx       # Entry point
```

## Best Practices

- Use TypeScript for type safety
- Follow the component structure for new features
- Keep components small and focused
- Use Shadcn/UI components for consistency
- Add meaningful comments for complex logic

## License

This project is licensed under the MIT License - see the LICENSE file in the root directory for details.
