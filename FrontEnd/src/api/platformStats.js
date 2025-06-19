// src/api/platformStats.js
import { ElMessage } from 'element-plus'

/**
 * 获取平台统计数据
 * @returns {Promise<Object>} 平台统计数据
 */
export const fetchStats = async () => {
  try {
    const response = await fetch('/api/platform-stats')
    if (!response.ok) throw new Error('获取数据失败')
    return await response.json()
  } catch (error) {
    console.error('Error fetching stats:', error)
    ElMessage.error('获取平台数据失败')
    throw error // 抛出错误，由调用方处理
  }
}

/**
 * 触发模型训练任务
 * @returns {Promise<Object>} 后端响应结果
 */
export const triggerTraining = async () => {
  try {
    const response = await fetch('/api/trigger-training', {
      method: 'POST'
    })
    if (!response.ok) throw new Error('触发训练失败')
    const data = await response.json()
    ElMessage.success(data.message)
    return data
  } catch (error) {
    console.error('Error triggering training:', error)
    ElMessage.error('触发训练失败')
    throw error
  }
}