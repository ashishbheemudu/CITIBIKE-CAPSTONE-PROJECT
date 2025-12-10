import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import SystemOverview from './pages/SystemOverview';

import MapExplorer from './pages/MapExplorer';

import RouteAnalysis from './pages/RouteAnalysis';

import StationDrilldown from './pages/StationDrilldown';

import Prediction from './pages/Prediction';

import AdvancedAnalytics from './pages/AdvancedAnalytics';

import RebalancingDashboard from './pages/RebalancingDashboard';
import EquityHeatmap from './pages/EquityHeatmap';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<SystemOverview />} />
          <Route path="/map" element={<MapExplorer />} />
          <Route path="/routes" element={<RouteAnalysis />} />
          <Route path="/stations" element={<StationDrilldown />} />
          <Route path="/prediction" element={<Prediction />} />
          <Route path="/advanced" element={<AdvancedAnalytics />} />
          <Route path="/rebalancing" element={<RebalancingDashboard />} />
          <Route path="/equity" element={<EquityHeatmap />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
