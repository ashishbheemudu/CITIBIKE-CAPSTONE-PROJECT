import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        this.setState({ error, errorInfo });
        console.error("Uncaught error:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="p-8 bg-background text-foreground h-screen overflow-auto">
                    <h1 className="text-2xl font-bold text-destructive mb-4">Something went wrong.</h1>
                    <div className="bg-secondary/30 p-4 rounded-lg border border-border mb-4">
                        <h2 className="font-mono font-bold text-lg mb-2">{this.state.error && this.state.error.toString()}</h2>
                        <pre className="font-mono text-xs whitespace-pre-wrap text-muted-foreground">
                            {this.state.errorInfo && this.state.errorInfo.componentStack}
                        </pre>
                    </div>
                    <button
                        onClick={() => window.location.reload()}
                        className="px-4 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90"
                    >
                        Reload Page
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
