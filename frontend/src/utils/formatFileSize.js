/**
 * Formats file size from bytes to human-readable format
 * @param {number|string} bytes - File size in bytes
 * @returns {string} Formatted file size (e.g., "1.2 KB", "3.4 MB")
 */
export const formatFileSize = (bytes) => {
  // Convert to number if it's a string, handle invalid values
  const numBytes = Number(bytes);
  
  if (isNaN(numBytes) || numBytes < 0) {
    return '0 B';
  }
  
  if (numBytes === 0) {
    return '0 B';
  }
  
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const k = 1024;
  const decimals = 1;
  
  const i = Math.floor(Math.log(numBytes) / Math.log(k));
  const size = numBytes / Math.pow(k, i);
  
  return `${size.toFixed(decimals)} ${units[i]}`;
};