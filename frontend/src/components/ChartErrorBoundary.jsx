import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

class ChartErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        console.error('Chart rendering error:', error, errorInfo);
    }

    handleReset = () => {
        this.setState({ hasError: false, error: null });
    };

    render() {
        if (this.state.hasError) {
            return (
                <div className="flex flex-col items-center justify-center h-full min-h-[200px] bg-secondary/10 rounded-lg p-6">
                    <AlertCircle className="w-12 h-12 text-destructive mb-3" />
                    <h3 className="text-lg font-semibold text-foreground mb-2">
                        Chart Error
                    </h3>
                    <p className="text-sm text-muted-foreground text-center mb-4">
                        Unable to render this chart. The data may be invalid or missing.
                    </p>
                    <button
                        onClick={this.handleReset}
                        className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
                    >
                        <RefreshCw className="w-4 h-4" />
                        Try Again
                    </button>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ChartErrorBoundary;
