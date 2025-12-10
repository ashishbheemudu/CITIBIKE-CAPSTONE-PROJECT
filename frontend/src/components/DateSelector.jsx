import React from 'react';
import { ChevronDown } from 'lucide-react';

const DateSelector = ({ value, onChange, className = "" }) => {
    // Value is expected to be YYYY-MM-DD string
    const date = value ? new Date(value) : new Date();

    // Helper to update date
    const updateDate = (type, val) => {
        const newDate = new Date(date);
        if (type === 'year') newDate.setFullYear(parseInt(val));
        if (type === 'month') newDate.setMonth(parseInt(val));
        if (type === 'day') newDate.setDate(parseInt(val));

        // Format to YYYY-MM-DD
        const y = newDate.getFullYear();
        const m = String(newDate.getMonth() + 1).padStart(2, '0');
        const d = String(newDate.getDate()).padStart(2, '0');
        onChange(`${y}-${m}-${d}`);
    };

    const years = Array.from({ length: 14 }, (_, i) => 2013 + i).reverse(); // 2013-2026
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const days = Array.from({ length: 31 }, (_, i) => i + 1);

    return (
        <div className={`grid grid-cols-3 gap-2 ${className}`}>
            {/* Year */}
            <div className="relative group">
                <select
                    value={date.getFullYear()}
                    onChange={(e) => updateDate('year', e.target.value)}
                    className="w-full bg-[#0a0b14] border border-white/10 rounded-lg px-2 py-2 text-xs text-white focus:outline-none focus:border-indigo-500/50 appearance-none cursor-pointer hover:bg-white/5 transition-colors"
                >
                    {years.map(year => (
                        <option key={year} value={year}>{year}</option>
                    ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-white/30 pointer-events-none" />
            </div>

            {/* Month */}
            <div className="relative group">
                <select
                    value={date.getMonth()}
                    onChange={(e) => updateDate('month', e.target.value)}
                    className="w-full bg-[#0a0b14] border border-white/10 rounded-lg px-2 py-2 text-xs text-white focus:outline-none focus:border-indigo-500/50 appearance-none cursor-pointer hover:bg-white/5 transition-colors"
                >
                    {months.map((m, i) => (
                        <option key={i} value={i}>{m}</option>
                    ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-white/30 pointer-events-none" />
            </div>

            {/* Day */}
            <div className="relative group">
                <select
                    value={date.getDate()}
                    onChange={(e) => updateDate('day', e.target.value)}
                    className="w-full bg-[#0a0b14] border border-white/10 rounded-lg px-2 py-2 text-xs text-white focus:outline-none focus:border-indigo-500/50 appearance-none cursor-pointer hover:bg-white/5 transition-colors"
                >
                    {days.map(day => (
                        <option key={day} value={day}>{day}</option>
                    ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-white/30 pointer-events-none" />
            </div>
        </div>
    );
};

export default DateSelector;
