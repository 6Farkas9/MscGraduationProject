// setup-frontend-structure.js
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// 获取 __dirname 的ESM等效方式
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = process.cwd();

// 需要创建的目录结构
const directories = [
  // assets
  'src/assets/css',
  'src/assets/images',
  
  // components
  'src/components/common',
  'src/components/charts',
  'src/components/learning',
  
  // views
  'src/views/dashboard',
  'src/views/analysis',
  'src/views/management',
  
  // stores
  'src/stores',
  
  // services
  'src/services/api',
  'src/services/types',
  
  // utils
  'src/utils',
  
  // router
  'src/router',
  
  // types
  'types'
];

// 需要创建的文件列表（空文件占位）
const files = [
  // assets
  'src/assets/css/global.css',
  'src/assets/css/variables.css',
  
  // components/common
  'src/components/common/AppHeader.vue',
  'src/components/common/AppSidebar.vue',
  'src/components/common/LoadingSpinner.vue',
  'src/components/common/DataCard.vue',
  
  // components/charts
  'src/components/charts/RadarChart.vue',
  'src/components/charts/LineChart.vue',
  'src/components/charts/BarChart.vue',
  'src/components/charts/KnowledgeGraph.vue',
  
  // components/learning
  'src/components/learning/BehaviorTable.vue',
  'src/components/learning/PreferenceTags.vue',
  'src/components/learning/ResourceList.vue',
  
  // views/dashboard
  'src/views/dashboard/LearnerDashboard.vue',
  'src/views/dashboard/SystemOverview.vue',
  'src/views/dashboard/ManagementPanel.vue',
  
  // views/analysis
  'src/views/analysis/CognitiveAnalysis.vue',
  'src/views/analysis/SocialAnalysis.vue',
  'src/views/analysis/BehaviorAnalysis.vue',
  
  // views/management
  'src/views/management/UserManagement.vue',
  'src/views/management/DataManagement.vue',
  'src/views/management/ModelManagement.vue',
  
  // stores
  'src/stores/index.ts',
  'src/stores/userStore.ts',
  'src/stores/systemStore.ts',
  'src/stores/analysisStore.ts',
  'src/stores/simulationStore.ts',
  
  // services/api
  'src/services/api/index.ts',
  'src/services/api/userApi.ts',
  'src/services/api/analysisApi.ts',
  'src/services/api/systemApi.ts',
  'src/services/api/simulationApi.ts',
  
  // services/types
  'src/services/types/user.ts',
  'src/services/types/learning.ts',
  'src/services/types/analysis.ts',
  'src/services/types/common.ts',
  
  // utils
  'src/utils/http.ts',
  'src/utils/formatters.ts',
  'src/utils/validators.ts',
  'src/utils/constants.ts',
  
  // router
  'src/router/index.ts',
  
  // types
  'types/coi.ts',
  'types/learning.ts',
  'types/api.ts',
  
  // 其他文件
  'src/App.vue',
  'src/main.ts'
];

// 创建目录的函数
function createDirectories() {
  console.log('🚀 开始创建目录结构...\n');
  
  directories.forEach(dir => {
    const fullPath = path.join(projectRoot, dir);
    if (!fs.existsSync(fullPath)) {
      fs.mkdirSync(fullPath, { recursive: true });
      console.log(`✅ 创建目录: ${dir}`);
    } else {
      console.log(`📁 目录已存在: ${dir}`);
    }
  });
  
  console.log('\n📁 目录结构创建完成!\n');
}

// 创建文件的函数
function createFiles() {
  console.log('📄 开始创建文件...\n');
  
  files.forEach(file => {
    const fullPath = path.join(projectRoot, file);
    if (!fs.existsSync(fullPath)) {
      // 根据文件类型创建不同的初始内容
      let content = '';
      
      if (file.endsWith('.vue')) {
        content = `<!-- ${path.basename(file)} -->\n<template>\n  <div>\n    <h2>${path.basename(file, '.vue')}</h2>\n    <p>组件开发中...</p>\n  </div>\n</template>\n\n<script setup lang="ts">\n// TODO: 实现 ${path.basename(file, '.vue')} 组件\n</script>\n\n<style scoped>\n/* 样式待添加 */\n</style>`;
      } else if (file.endsWith('.ts') || file.endsWith('.js')) {
        content = `// ${path.basename(file)}\n// TODO: 实现 ${path.basename(file)} 功能\nexport {};`;
      } else if (file.endsWith('.css')) {
        content = `/* ${path.basename(file)} */\n/* 样式文件待完善 */`;
      } else {
        content = `# ${path.basename(file)}\n\n文件内容待完善`;
      }
      
      fs.writeFileSync(fullPath, content);
      console.log(`✅ 创建文件: ${file}`);
    } else {
      console.log(`📄 文件已存在: ${file}`);
    }
  });
  
  console.log('\n📄 文件创建完成!\n');
}

// 更新 package.json 添加必要的脚本
function updatePackageJson() {
  const packageJsonPath = path.join(projectRoot, 'package.json');
  
  if (fs.existsSync(packageJsonPath)) {
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    
    // 添加有用的脚本
    if (!packageJson.scripts) {
      packageJson.scripts = {};
    }
    
    packageJson.scripts['setup'] = 'node setup-frontend-structure.js';
    packageJson.scripts['dev'] = 'vite';
    packageJson.scripts['build'] = 'vue-tsc && vite build';
    packageJson.scripts['preview'] = 'vite preview'; // 这里修复了变量名错误
    
    fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
    console.log('✅ 更新 package.json 脚本');
  }
}

// 创建 README.md 文件
function createReadme() {
  const readmePath = path.join(projectRoot, 'FRONTEND_README.md');
  const content = `# 元宇宙学习者服务系统 - 前端

## 项目结构

\`\`\`
frontend/
├── src/
│   ├── assets/                 # 静态资源
│   ├── components/             # 通用组件
│   ├── views/                  # 页面视图
│   ├── stores/                 # 状态管理
│   ├── services/               # API服务
│   ├── utils/                  # 工具函数
│   ├── router/                 # 路由配置
│   └── types/                  # 类型定义
└── types/                      # 全局类型
\`\`\`

## 功能模块

### 1. 学习者个人画像展示
- 基本信息面板
- 能力预测图表 (KT/CD)
- COI分析展示
- 行为记录表格
- 偏好分析和推荐资源

### 2. 系统信息概览
- 数据统计卡片
- 系统状态监控
- 模型管理

### 3. 模拟功能
- 学习者初始化
- 行为模拟

## 开发命令

\`\`\`bash
# 安装依赖
npm install

# 开发模式
npm run dev

# 构建项目
npm run build

# 预览构建结果
npm run preview
\`\`\`

## 技术栈

- Vue 3 + Composition API
- TypeScript
- Vite
- Pinia (状态管理)
- Element Plus (UI组件)
- ECharts (图表)
- Axios (HTTP请求)

## 后端接口

所有API接口定义在 \`src/services/api/\` 目录中，待后端开发完成后更新对应的URL和实现。
`;

  fs.writeFileSync(readmePath, content);
  console.log('✅ 创建 FRONTEND_README.md');
}

// 主函数
function main() {
  console.log('🎯 开始设置前端项目结构...\n');
  
  try {
    // 检查是否在正确的目录
    if (!fs.existsSync(path.join(projectRoot, 'package.json'))) {
      console.error('❌ 错误: 请在项目根目录运行此脚本');
      process.exit(1);
    }
    
    createDirectories();
    createFiles();
    updatePackageJson();
    createReadme();
    
    console.log('🎉 前端项目结构设置完成!');
    console.log('\n📋 下一步:');
    console.log('1. 运行 npm install 安装依赖');
    console.log('2. 运行 npm run dev 启动开发服务器');
    console.log('3. 开始逐个文件实现功能');
    
  } catch (error) {
    console.error('❌ 设置过程中发生错误:', error);
    process.exit(1);
  }
}

// 运行主函数
main();