# 开发知识

[知识库主页](../README.md) · [业务知识](business.md) · [全部文档与附档](../catalog.md)

先理解系统职责和业务对象，再进入具体代码。工程任务仍先读[AGENTS](../../../AGENTS.md)与原Hermes；知识库帮助解释与取证，不替代模块路由、专业契约或授权。

## 建立完整技术图景

1. [系统怎样分层，能力属于谁？](../topics/developer-architecture.md)：浏览器、API、领域服务、计算、持久化及独立项目
2. [数据怎样进入研究与计算？](../topics/developer-data-compute.md)：采集/ETL/候选/激活、PIT、类型/时点/图、NJIT和不可变成果
3. [模块、页面和交接如何保持一致？](../topics/developer-modules.md)：研究能力群、共享导航、URL/冻结关联、前端状态及助手挂载
4. [关键架构决定为何如此？](../topics/developer-decisions.md)：单一当前实现、存储解耦、公共图、预热门禁和Portable演进
5. [一次开发怎样达到验收和交付条件？](../topics/developer-delivery.md)：授权范围、计算/前端/数据专业检查、文档、PR和四轴证据
6. [怎样运行系统，生产还需要哪些独立条件？](../topics/developer-operations.md)：依赖、存储、就绪、同源部署、身份/网关/制品和生产迁移

## 从全景进入具体工作

原[文档索引](../../README.md)和[生成目录](../catalog.md#模块归属)包含全部36个当前Hermes模块的原归属，实际路径/测试/命令仍读取原repo_map。不要在Wiki摘要里维护第二份可执行清单。

外部方案和竞品架构走[研究资料工作流](../topics/research-library.md)：观察到的接口/行为与推测的内部实现分开。局部C++适配不等于全系统已迁移；静态代码、历史截图和旧测试不能代替当前运行验证。
