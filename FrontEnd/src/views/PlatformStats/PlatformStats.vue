<template>
  <div class="platformstats">
    <div class="header">
      <h1>学习平台数据概览</h1>
      <p class="subtitle">冰冻三尺，非一日之寒，成大事者不拘小节</p>
      <p class="author">—— Mr. Xie</p>
    </div>

    <div class="data-cards">
      <el-row :gutter="20">
        <el-col :xs="24" :sm="12" :lg="6">
          <el-card class="data-card">
            <div class="card-content">
              <div class="card-icon">
                <el-icon><Collection /></el-icon>
              </div>
              <div class="card-text">
                <h3>领域数量</h3>
                <p class="value">{{ stats.domainCount }}</p>
                <p class="description">涵盖的学习领域总数</p>
              </div>
            </div>
          </el-card>
        </el-col>
        
        <el-col :xs="24" :sm="12" :lg="6">
          <el-card class="data-card">
            <div class="card-content">
              <div class="card-icon">
                <el-icon><User /></el-icon>
              </div>
              <div class="card-text">
                <h3>学习者数量</h3>
                <p class="value">{{ stats.learnerCount }}</p>
                <p class="description">活跃学习者总数</p>
              </div>
            </div>
          </el-card>
        </el-col>
        
        <el-col :xs="24" :sm="12" :lg="6">
          <el-card class="data-card">
            <div class="card-content">
              <div class="card-icon">
                <el-icon><Opportunity /></el-icon>
              </div>
              <div class="card-text">
                <h3>场景数量</h3>
                <p class="value">{{ stats.scenarioCount }}</p>
                <p class="description">学习场景总数</p>
              </div>
            </div>
          </el-card>
        </el-col>
        
        <el-col :xs="24" :sm="12" :lg="6">
          <el-card class="data-card">
            <div class="card-content">
              <div class="card-icon">
                <el-icon><Reading /></el-icon>
              </div>
              <div class="card-text">
                <h3>知识点数量</h3>
                <p class="value">{{ stats.knowledgePointCount }}</p>
                <p class="description">系统知识点总数</p>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <div class="model-info">
      <el-card>
        <div class="model-content">
          <h3>后台深度学习模型状态</h3>
          <p>上次训练时间: {{ stats.lastTrainingTime }}</p>
          <p>模型版本: {{ stats.modelVersion }}</p>
          <el-progress :percentage="stats.accuracy" :color="customColors"></el-progress>
          <el-button type="primary" class="train-btn" @click="triggerTraining">手动触发训练</el-button>
        </div>
      </el-card>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import {
  Collection,
  User,
  Opportunity,
  Reading
} from '@element-plus/icons-vue'

// 模拟数据
const stats = ref({
  domainCount: 12,
  learnerCount: 8562,
  scenarioCount: 34,
  knowledgePointCount: 1258,
  lastTrainingTime: '2023-11-15 14:30:22',
  modelVersion: 'v2.3.1',
  accuracy: 87.5
})

const customColors = [
  { color: '#f56c6c', percentage: 20 },
  { color: '#e6a23c', percentage: 40 },
  { color: '#5cb87a', percentage: 60 },
  { color: '#1989fa', percentage: 80 },
  { color: '#6f7ad3', percentage: 100 }
]

const triggerTraining = () => {
  ElMessage.info('已触发模型训练任务')
}
</script>

<style scoped>
.dashboard-container {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.header h1 {
  font-size: 28px;
  color: #333;
  margin-bottom: 10px;
}

.subtitle {
  font-size: 16px;
  color: #666;
  font-style: italic;
  margin-bottom: 5px;
}

.author {
  font-size: 14px;
  color: #999;
  text-align: right;
  margin-right: 20px;
}

.data-cards {
  margin-bottom: 20px;
}

.data-card {
  margin-bottom: 20px;
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.card-content {
  display: flex;
  align-items: center;
  padding: 15px;
}

.card-icon {
  font-size: 36px;
  width: 60px;
  height: 60px;
  line-height: 60px;
  text-align: center;
  background: #f0f7ff;
  border-radius: 50%;
  color: #409eff;
  margin-right: 15px;
}

.card-text h3 {
  font-size: 16px;
  color: #666;
  margin: 0 0 5px 0;
}

.card-text .value {
  font-size: 24px;
  font-weight: bold;
  color: #333;
  margin: 0 0 5px 0;
}

.card-text .description {
  font-size: 12px;
  color: #999;
  margin: 0;
}

.model-info {
  margin-bottom: 20px;
}

.model-content {
  padding: 15px;
}

.model-content h3 {
  margin-top: 0;
  color: #333;
}

.model-content p {
  margin: 10px 0;
  color: #666;
}

.train-btn {
  margin-top: 15px;
}

@media (max-width: 768px) {
  .card-content {
    flex-direction: column;
    text-align: center;
  }
  
  .card-icon {
    margin-right: 0;
    margin-bottom: 10px;
  }
}
</style>