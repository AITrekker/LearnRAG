import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Home, Zap, Search } from 'lucide-react';
import clsx from 'clsx';

const Sidebar = () => {
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: Home },
    { name: 'Embeddings', href: '/embeddings', icon: Zap },
    { name: 'Search', href: '/search', icon: Search },
  ];

  return (
    <motion.div
      initial={{ x: -250, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 bg-white shadow-lg border-r border-gray-200"
    >
      <nav className="p-4">
        <ul className="space-y-2">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href;
            return (
              <li key={item.name}>
                <Link
                  to={item.href}
                  className={clsx(
                    'flex items-center px-3 py-2 rounded-lg transition-all duration-200',
                    isActive
                      ? 'bg-primary-50 text-primary-700 border-r-2 border-primary-700'
                      : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                  )}
                >
                  <motion.div
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <item.icon className={clsx(
                      'w-5 h-5 mr-3',
                      isActive ? 'text-primary-600' : 'text-gray-400'
                    )} />
                  </motion.div>
                  <span className="font-medium">{item.name}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
    </motion.div>
  );
};

export default Sidebar;