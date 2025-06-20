// @/api/learner.js

/**
 * 获取学习者基本信息
 * @param {string} uid 学习者ID
 * @returns {Promise} 包含学习者信息的Promise
 */
export function getLearnerInfo(uid) {
  return fetch(`/api/learner-info/${uid}`)
    .then(response => {
      if (!response.ok) {
        throw new Error('网络响应不正常');
      }
      return response.json();
    })
    .then(data => {
      return {
        code: data.code,
        message: data.message,
        data: data.data
      };
    })
    .catch(error => {
      console.error('获取学习者信息失败:', error);
      return {
        code: 500,
        message: '获取学习者信息失败',
        data: null
      };
    });
}

/**
 * 获取推荐内容
 * @param {string} uid 学习者ID
 * @returns {Promise} 包含推荐内容的Promise
 */
export function getRecommendations(uid) {
  return fetch(`/api/recommendations/${uid}`)
    .then(response => {
      if (!response.ok) {
        throw new Error('网络响应不正常');
      }
      return response.json();
    })
    .then(data => {
      return {
        code: data.code,
        message: data.message,
        data: data.data
      };
    })
    .catch(error => {
      console.error('获取推荐内容失败:', error);
      return {
        code: 500,
        message: '获取推荐内容失败',
        data: null
      };
    });
}