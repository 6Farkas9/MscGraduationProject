import mysql.connector
from pymongo import MongoClient
from collections import Counter, defaultdict
import datetime
from typing import Dict, List, Any, Tuple
import numpy as np

class LearnerBehaviorAnalyzer:
    """学习者行为分析器 - 基于xAPI数据"""
    
    def __init__(self):
        # 数据库连接配置
        self.mongo_client = MongoClient('mongodb://localhost:27017/')
        self.mongo_db = self.mongo_client['MLS']
        self.interaction_collection = self.mongo_db['Interaction']
        self.profile_collection = self.mongo_db['LearnerProfile']
        
        # 动词分类定义
        self.verb_categories = {
            'content_consumption': [
                'started', 'watched', 'paused', 'resumed', 'rewound', 'skipped', 'completed',
                'entered', 'explored', 'navigated', 'observed', 'interacted', 
                'activated', 'scanned', 'augmented', 'manipulated',
                'attempted', 'operated', 'solved',
                'joined', 'coordinated', 'collaborated', 'contributed',
                'experienced'
            ],
            'knowledge_construction': [
                'created_note', 'edited_note', 'highlighted', 'bookmarked', 'reviewed_notes',
                'added_concept', 'linked_concepts', 'reorganized_map', 'reviewed_map',
                'searched', 'refined_search', 'clicked_result'
            ],
            'problem_solving': [
                'attempted', 'solved', 'failed', 'retried', 'reviewed_solution',
                'started_test', 'answered', 'reviewed_answer', 'completed_test'
            ],
            'social_collaboration': [
                'posted_message', 'replied', 'liked_post', 'shared_resource',
                'gave_feedback', 'received_feedback', 'requested_help',
                'suggested_plan', 'assigned_task', 'reported_progress'
            ],
            'metacognition': [
                'set_goal', 'adjusted_goal', 'planned_schedule',
                'checked_progress', 'evaluated_understanding', 'identified_gap',
                'reflected_learning', 'identified_strength', 'noted_improvement'
            ]
        }
        
        # COI分析维度
        self.coi_dimensions = {
            'cognitive_presence': ['knowledge_construction', 'problem_solving', 'metacognition'],
            'social_presence': ['social_collaboration'],
            'content_engagement': ['content_consumption']
        }

    def get_learner_behavior_sequence(self, learner_uid: str) -> List[Dict]:
        """
        获取指定学习者的完整行为序列
        
        Args:
            learner_uid: 学习者唯一标识
            
        Returns:
            按时间排序的行为序列列表
        """
        print(f"📊 获取学习者 {learner_uid} 的行为序列...")
        
        # 从MongoDB查询学习者的所有行为记录
        behaviors = list(self.interaction_collection.find(
            {"actor.account.name": learner_uid}
        ).sort("timestamp", 1))  # 按时间升序排序
        
        print(f"✅ 找到 {len(behaviors)} 条行为记录")
        return behaviors

    def get_learner_profile(self, learner_uid: str) -> Dict:
        """
        获取学习者人设信息
        
        Args:
            learner_uid: 学习者唯一标识
            
        Returns:
            学习者人设字典
        """
        profile = self.profile_collection.find_one({"learner_uid": learner_uid})
        if not profile:
            print(f"⚠️  未找到学习者 {learner_uid} 的人设信息")
            return {}
        
        print(f"✅ 加载学习者人设: {profile.get('basic_info', {}).get('name', '未知')}")
        return profile

    def analyze_verb_distribution(self, behaviors: List[Dict]) -> Dict:
        """
        分析动词分布情况
        
        Args:
            behaviors: 行为序列
            
        Returns:
            动词分布统计字典
        """
        print("\n🔍 分析动词分布...")
        
        # 提取所有动词
        verbs = [behavior['verb']['display']['en-US'] for behavior in behaviors]
        verb_counter = Counter(verbs)
        
        # 按类别分类统计
        category_stats = {}
        for category, category_verbs in self.verb_categories.items():
            category_count = sum(verb_counter.get(verb, 0) for verb in category_verbs)
            category_stats[category] = {
                'total': category_count,
                'percentage': (category_count / len(verbs)) * 100 if verbs else 0,
                'verbs': {verb: verb_counter[verb] for verb in category_verbs if verb in verb_counter}
            }
        
        # 最常用动词
        top_verbs = verb_counter.most_common(10)
        
        return {
            'total_behaviors': len(verbs),
            'unique_verbs': len(verb_counter),
            'verb_counter': dict(verb_counter),
            'category_stats': category_stats,
            'top_verbs': top_verbs
        }

    def analyze_cognitive_presence(self, behaviors: List[Dict], verb_stats: Dict) -> Dict:
        """
        分析认知存在维度
        
        Args:
            behaviors: 行为序列
            verb_stats: 动词统计信息
            
        Returns:
            认知存在分析结果
        """
        print("\n🧠 分析认知存在...")
        
        # 获取认知相关行为
        cognitive_behaviors = [
            behavior for behavior in behaviors 
            if behavior['verb']['display']['en-US'] in 
            self.verb_categories['knowledge_construction'] + 
            self.verb_categories['problem_solving'] + 
            self.verb_categories['metacognition']
        ]
        
        # 知识建构深度分析
        knowledge_verbs = self.verb_categories['knowledge_construction']
        knowledge_behaviors = [b for b in cognitive_behaviors if b['verb']['display']['en-US'] in knowledge_verbs]
        
        # 问题解决能力分析
        problem_verbs = self.verb_categories['problem_solving']
        problem_behaviors = [b for b in cognitive_behaviors if b['verb']['display']['en-US'] in problem_verbs]
        
        # 元认知水平分析
        meta_verbs = self.verb_categories['metacognition']
        meta_behaviors = [b for b in cognitive_behaviors if b['verb']['display']['en-US'] in meta_verbs]
        
        # 计算认知活跃度
        cognitive_activity_ratio = len(cognitive_behaviors) / len(behaviors) if behaviors else 0
        
        return {
            'total_cognitive_behaviors': len(cognitive_behaviors),
            'cognitive_activity_ratio': cognitive_activity_ratio,
            'knowledge_construction': {
                'total': len(knowledge_behaviors),
                'note_taking': verb_stats['category_stats']['knowledge_construction']['verbs'].get('created_note', 0) +
                              verb_stats['category_stats']['knowledge_construction']['verbs'].get('edited_note', 0),
                'concept_mapping': verb_stats['category_stats']['knowledge_construction']['verbs'].get('linked_concepts', 0) +
                                  verb_stats['category_stats']['knowledge_construction']['verbs'].get('added_concept', 0),
                'search_behavior': verb_stats['category_stats']['knowledge_construction']['verbs'].get('searched', 0) +
                                  verb_stats['category_stats']['knowledge_construction']['verbs'].get('refined_search', 0)
            },
            'problem_solving': {
                'total': len(problem_behaviors),
                'success_rate': self._calculate_success_rate(problem_behaviors),
                'persistence': verb_stats['category_stats']['problem_solving']['verbs'].get('retried', 0),
                'review_behavior': verb_stats['category_stats']['problem_solving']['verbs'].get('reviewed_solution', 0)
            },
            'metacognition': {
                'total': len(meta_behaviors),
                'planning': verb_stats['category_stats']['metacognition']['verbs'].get('set_goal', 0) +
                           verb_stats['category_stats']['metacognition']['verbs'].get('planned_schedule', 0),
                'monitoring': verb_stats['category_stats']['metacognition']['verbs'].get('checked_progress', 0) +
                             verb_stats['category_stats']['metacognition']['verbs'].get('evaluated_understanding', 0),
                'reflection': verb_stats['category_stats']['metacognition']['verbs'].get('reflected_learning', 0) +
                             verb_stats['category_stats']['metacognition']['verbs'].get('identified_strength', 0)
            }
        }

    def analyze_social_presence(self, behaviors: List[Dict], verb_stats: Dict) -> Dict:
        """
        分析社会存在维度
        
        Args:
            behaviors: 行为序列
            verb_stats: 动词统计信息
            
        Returns:
            社会存在分析结果
        """
        print("\n👥 分析社会存在...")
        
        # 获取社交相关行为
        social_behaviors = [
            behavior for behavior in behaviors 
            if behavior['verb']['display']['en-US'] in self.verb_categories['social_collaboration']
        ]
        
        # 讨论参与度分析
        discussion_verbs = ['posted_message', 'replied', 'liked_post']
        discussion_count = sum(verb_stats['category_stats']['social_collaboration']['verbs'].get(verb, 0) 
                              for verb in discussion_verbs)
        
        # 反馈交流分析
        feedback_verbs = ['gave_feedback', 'received_feedback', 'requested_help']
        feedback_count = sum(verb_stats['category_stats']['social_collaboration']['verbs'].get(verb, 0) 
                            for verb in feedback_verbs)
        
        # 协作协调分析
        coordination_verbs = ['suggested_plan', 'assigned_task', 'reported_progress']
        coordination_count = sum(verb_stats['category_stats']['social_collaboration']['verbs'].get(verb, 0) 
                                for verb in coordination_verbs)
        
        # 计算社交活跃度
        social_activity_ratio = len(social_behaviors) / len(behaviors) if behaviors else 0
        
        return {
            'total_social_behaviors': len(social_behaviors),
            'social_activity_ratio': social_activity_ratio,
            'discussion_participation': discussion_count,
            'feedback_exchange': feedback_count,
            'coordination_behavior': coordination_count,
            'resource_sharing': verb_stats['category_stats']['social_collaboration']['verbs'].get('shared_resource', 0)
        }

    def analyze_content_engagement(self, behaviors: List[Dict], verb_stats: Dict) -> Dict:
        """
        分析内容参与度
        
        Args:
            behaviors: 行为序列
            verb_stats: 动词统计信息
            
        Returns:
            内容参与度分析结果
        """
        print("\n📚 分析内容参与度...")
        
        # 获取内容消费行为
        content_behaviors = [
            behavior for behavior in behaviors 
            if behavior['verb']['display']['en-US'] in self.verb_categories['content_consumption']
        ]
        
        # 学习深度分析（通过细致操作判断）
        deep_engagement_verbs = ['paused', 'resumed', 'rewound', 'navigated', 'observed', 'explored']
        deep_engagement_count = sum(verb_stats['verb_counter'].get(verb, 0) for verb in deep_engagement_verbs)
        
        # 学习完成度分析
        completion_verbs = ['completed', 'solved']
        completion_count = sum(verb_stats['verb_counter'].get(verb, 0) for verb in completion_verbs)
        
        # 跳过行为分析
        skip_count = verb_stats['verb_counter'].get('skipped', 0)
        
        # 计算内容参与度
        content_engagement_ratio = len(content_behaviors) / len(behaviors) if behaviors else 0
        
        return {
            'total_content_behaviors': len(content_behaviors),
            'content_engagement_ratio': content_engagement_ratio,
            'deep_engagement_indicators': deep_engagement_count,
            'completion_rate': completion_count,
            'skip_behavior': skip_count,
            'engagement_depth': (deep_engagement_count / len(content_behaviors)) if content_behaviors else 0
        }

    def determine_coi_type(self, cognitive_analysis: Dict, social_analysis: Dict, 
                          content_analysis: Dict, profile: Dict) -> Dict:
        """
        确定学习者的COI类型
        
        Args:
            cognitive_analysis: 认知存在分析结果
            social_analysis: 社会存在分析结果  
            content_analysis: 内容参与度分析结果
            profile: 学习者人设
            
        Returns:
            COI类型分析结果
        """
        print("\n🎯 确定COI类型...")
        
        # 认知存在水平评估
        cognitive_level = self._assess_cognitive_level(cognitive_analysis)
        
        # 社会存在水平评估
        social_level = self._assess_social_level(social_analysis)
        
        # 内容参与度评估
        engagement_level = self._assess_engagement_level(content_analysis)
        
        # 确定COI类型
        coi_type = self._classify_coi_type(cognitive_level, social_level, engagement_level)
        
        # 与预设人设对比
        profile_consistency = self._check_profile_consistency(coi_type, profile)
        
        return {
            'coi_type': coi_type,
            'cognitive_level': cognitive_level,
            'social_level': social_level,
            'engagement_level': engagement_level,
            'profile_consistency': profile_consistency,
            'detailed_assessment': {
                'cognitive_presence': cognitive_level,
                'social_presence': social_level,
                'content_engagement': engagement_level
            }
        }

    def _calculate_success_rate(self, behaviors: List[Dict]) -> float:
        """计算行为成功率"""
        if not behaviors:
            return 0.0
        
        success_count = sum(1 for behavior in behaviors 
                          if behavior.get('result', {}).get('success', False))
        return success_count / len(behaviors)

    def _assess_cognitive_level(self, analysis: Dict) -> str:
        """评估认知存在水平"""
        cognitive_ratio = analysis['cognitive_activity_ratio']
        knowledge_total = analysis['knowledge_construction']['total']
        meta_total = analysis['metacognition']['total']
        
        if cognitive_ratio > 0.4 and meta_total > 5:
            return "高认知存在"
        elif cognitive_ratio > 0.2 and knowledge_total > 3:
            return "中等认知存在"
        else:
            return "低认知存在"

    def _assess_social_level(self, analysis: Dict) -> str:
        """评估社会存在水平"""
        social_ratio = analysis['social_activity_ratio']
        discussion_count = analysis['discussion_participation']
        
        if social_ratio > 0.3 and discussion_count > 10:
            return "高社会存在"
        elif social_ratio > 0.1 and discussion_count > 3:
            return "中等社会存在"
        else:
            return "低社会存在"

    def _assess_engagement_level(self, analysis: Dict) -> str:
        """评估内容参与度水平"""
        engagement_ratio = analysis['content_engagement_ratio']
        engagement_depth = analysis['engagement_depth']
        
        if engagement_ratio > 0.5 and engagement_depth > 0.3:
            return "高内容参与度"
        elif engagement_ratio > 0.3:
            return "中等内容参与度"
        else:
            return "低内容参与度"

    def _classify_coi_type(self, cognitive_level: str, social_level: str, engagement_level: str) -> str:
        """分类COI类型"""
        if cognitive_level == "高认知存在" and social_level == "高社会存在":
            return "平衡发展型"
        elif cognitive_level == "高认知存在" and social_level in ["低社会存在", "中等社会存在"]:
            return "认知主导型"
        elif social_level == "高社会存在" and cognitive_level in ["低认知存在", "中等认知存在"]:
            return "社交主导型"
        elif engagement_level == "高内容参与度" and cognitive_level == "中等认知存在":
            return "积极参与型"
        else:
            return "观察学习型"

    def _check_profile_consistency(self, coi_type: str, profile: Dict) -> Dict:
        """检查与预设人设的一致性"""
        if not profile:
            return {"consistent": True, "message": "无人设信息可对比"}
        
        expected_style = profile.get('learning_style', '')
        social_engagement = profile.get('social_profile', {}).get('social_engagement', '')
        
        # 简单的一致性检查逻辑
        consistency_rules = {
            "平衡发展型": ["社交协作型", "混合适应型"],
            "认知主导型": ["视觉听觉型", "沉浸体验型", "动手实践型"],
            "社交主导型": ["社交协作型"],
            "积极参与型": ["混合适应型", "动手实践型"]
        }
        
        is_consistent = expected_style in consistency_rules.get(coi_type, [])
        
        return {
            "consistent": is_consistent,
            "expected_style": expected_style,
            "actual_coi_type": coi_type,
            "message": "符合预期" if is_consistent else "与预期人设存在差异"
        }

    def generate_comprehensive_report(self, learner_uid: str) -> Dict:
        """
        生成综合分析报告
        
        Args:
            learner_uid: 学习者唯一标识
            
        Returns:
            完整分析报告
        """
        print(f"\n{'='*60}")
        print(f"🎓 学习者行为分析报告 - {learner_uid}")
        print(f"{'='*60}")
        
        # 1. 获取基础数据
        behaviors = self.get_learner_behavior_sequence(learner_uid)
        profile = self.get_learner_profile(learner_uid)
        
        if not behaviors:
            print("❌ 无行为数据可分析")
            return {}
        
        # 2. 动词分布分析
        verb_analysis = self.analyze_verb_distribution(behaviors)
        
        # 3. 多维度深度分析
        cognitive_analysis = self.analyze_cognitive_presence(behaviors, verb_analysis)
        social_analysis = self.analyze_social_presence(behaviors, verb_analysis)
        content_analysis = self.analyze_content_engagement(behaviors, verb_analysis)
        
        # 4. COI类型判定
        coi_analysis = self.determine_coi_type(cognitive_analysis, social_analysis, content_analysis, profile)
        
        # 5. 生成报告
        report = {
            'learner_uid': learner_uid,
            'profile_info': profile,
            'basic_stats': {
                'total_behaviors': len(behaviors),
                'time_span': self._get_time_span(behaviors),
                'unique_verbs': verb_analysis['unique_verbs']
            },
            'verb_analysis': verb_analysis,
            'cognitive_analysis': cognitive_analysis,
            'social_analysis': social_analysis,
            'content_analysis': content_analysis,
            'coi_analysis': coi_analysis
        }
        
        # 6. 打印摘要报告
        self._print_summary_report(report)
        
        return report

    def _get_time_span(self, behaviors: List[Dict]) -> str:
        """计算行为时间跨度"""
        if not behaviors:
            return "无数据"
        
        timestamps = [behavior['timestamp'] for behavior in behaviors]
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        start_date = start_time[:10]  # 提取日期部分
        end_date = end_time[:10]
        
        return f"{start_date} 至 {end_date}"

    def _print_summary_report(self, report: Dict):
        """打印摘要报告"""
        print(f"\n{'='*60}")
        print("📈 分析报告摘要")
        print(f"{'='*60}")
        
        basic = report['basic_stats']
        verb = report['verb_analysis']
        coi = report['coi_analysis']
        
        print(f"📊 基础统计:")
        print(f"  总行为数: {basic['total_behaviors']}")
        print(f"  时间跨度: {basic['time_span']}")
        print(f"  使用动词种类: {basic['unique_verbs']}")
        
        print(f"\n🎯 COI分析结果:")
        print(f"  COI类型: {coi['coi_type']}")
        print(f"  认知存在: {coi['cognitive_level']}")
        print(f"  社会存在: {coi['social_level']}")
        print(f"  内容参与: {coi['engagement_level']}")
        print(f"  人设一致性: {coi['profile_consistency']['message']}")
        
        print(f"\n🔍 关键指标:")
        cognitive = report['cognitive_analysis']
        social = report['social_analysis']
        content = report['content_analysis']
        
        print(f"  认知活跃度: {cognitive['cognitive_activity_ratio']:.1%}")
        print(f"  知识建构行为: {cognitive['knowledge_construction']['total']} 次")
        print(f"  问题解决成功率: {cognitive['problem_solving']['success_rate']:.1%}")
        print(f"  社交参与度: {social['social_activity_ratio']:.1%}")
        print(f"  内容参与深度: {content['engagement_depth']:.1%}")

    def analyze_multiple_learners(self, learner_uids: List[str]) -> Dict:
        """
        批量分析多个学习者
        
        Args:
            learner_uids: 学习者UID列表
            
        Returns:
            批量分析结果
        """
        print(f"\n👥 开始批量分析 {len(learner_uids)} 个学习者...")
        
        results = {}
        for uid in learner_uids:
            print(f"\n--- 分析学习者 {uid} ---")
            try:
                results[uid] = self.generate_comprehensive_report(uid)
            except Exception as e:
                print(f"❌ 分析学习者 {uid} 时出错: {e}")
                results[uid] = {'error': str(e)}
        
        # 生成群体分析
        group_analysis = self._analyze_learner_group(results)
        
        return {
            'individual_results': results,
            'group_analysis': group_analysis
        }

    def _analyze_learner_group(self, results: Dict) -> Dict:
        """分析学习者群体特征"""
        valid_results = {k: v for k, v in results.items() if 'coi_analysis' in v}
        
        if not valid_results:
            return {}
        
        # COI类型分布
        coi_types = [result['coi_analysis']['coi_type'] for result in valid_results.values()]
        coi_distribution = Counter(coi_types)
        
        # 认知水平分布
        cognitive_levels = [result['coi_analysis']['cognitive_level'] for result in valid_results.values()]
        cognitive_distribution = Counter(cognitive_levels)
        
        return {
            'total_analyzed': len(valid_results),
            'coi_distribution': dict(coi_distribution),
            'cognitive_distribution': dict(cognitive_distribution),
            'social_distribution': dict(Counter([result['coi_analysis']['social_level'] for result in valid_results.values()]))
        }


def main():
    """主函数 - 使用示例"""
    analyzer = LearnerBehaviorAnalyzer()
    
    # 单个学习者分析
    learner_uid = "lrn_a53ab53ec7a54ae0aa6b515f527678f6"  # 替换为实际的学习者UID
    report = analyzer.generate_comprehensive_report(learner_uid)
    
    # 批量学习者分析示例
    # learner_uids = ["LRN001", "LRN002", "LRN003"]
    # batch_results = analyzer.analyze_multiple_learners(learner_uids)


if __name__ == "__main__":
    main()