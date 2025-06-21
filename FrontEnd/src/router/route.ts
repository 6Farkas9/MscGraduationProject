/***
 * @常量路由
 */
export const Layout = () => import("@/views/Layout/index.vue")
export const constRouter = [
  {
    path: "/",
    redirect: "/platformstats",
    name: "layout_platformstats",
    meta: { title: "", icon: "", isShow: false },
    component: Layout,
    children: [
      {
        path: "/platformstats",
        name: "platformstats",
        meta: { title: "平台状态", icon: "", isShow: true },
        component: () => import("@/views/PlatformStats/PlatformStats.vue"),
      },
    ],
  },
  {
    path: "/learnerinfo",
    name: "layout_lrninfo",
    meta: { title: "", icon: "", isShow: false },
    component: Layout,
    children: [
      {
        path: "/learnerinfo",
        name: "learnerinfo",
        meta: { title: "学习者数据", icon: "", isShow: true },
        component: () => import("@/views/LearnerInfo/LearnerInfo.vue"),
      },
    ],
  },
  // {
  //   path: "/home",
  //   // redirect: "/home",
  //   name: "layout_home",
  //   meta: { title: "", icon: "", isShow: false },
  //   component: Layout,
  //   children: [
  //     {
  //       path: "/home",
  //       name: "home",
  //       meta: { title: "首页", icon: "", isShow: true },
  //       component: () => import("@/views/Home/index.vue"),
  //     },
  //   ],
  // },
  // {
  //   path: "/components",
  //   name: "components",
  //   redirect: "/components/pagination",
  //   meta: { title: "封装组件", icon: "Menu", isShow: true },
  //   component: Layout,
  //   children: [
  //     {
  //       path: "/components/pagination",
  //       name: "pagination",
  //       meta: { title: "分页器 ", icon: "MoreFilled", isShow: true },
  //       component: () => import("@/views/Components/pagination/index.vue"),
  //     },
  //   ],
  // },

  // {
  //   path: "/moremenu",
  //   name: "moremenu",
  //   meta: { title: "多级菜单", icon: "Operation", isShow: true },
  //   component: Layout,
  //   children: [
  //     {
  //       path: "/moremenu/menu-one",
  //       name: "menu-one",
  //       meta: { title: "一级菜单 ", icon: "DArrowRight", isShow: true },
  //       component: () => import("@/views/MoreMenu/MenuOne/index.vue"),
  //       children: [
  //         {
  //           path: "/moremenu/menu-one/menu-two",
  //           name: "menu-two",
  //           meta: { title: "二级菜单 ", icon: "DArrowRight", isShow: true },
  //           component: () =>
  //             import("@/views/MoreMenu/MenuOne/MenuTwo/index.vue"),
  //           children: [
  //             {
  //               path: "/moremenu/menu-one/menu-two/menu-three-1",
  //               name: "menu-three-1",
  //               meta: {
  //                 title: "三级菜单-1",
  //                 icon: "DArrowRight",
  //                 isShow: true,
  //               },
  //               component: () =>
  //                 import(
  //                   "@/views/MoreMenu/MenuOne/MenuTwo/MenuThree/index-1.vue"
  //                 ),
  //             },
  //             {
  //               path: "/moremenu/menu-one/menu-two/menu-three-2",
  //               name: "menu-three-2",
  //               meta: {
  //                 title: "三级菜单-2",
  //                 icon: "DArrowRight",
  //                 isShow: true,
  //               },
  //               component: () =>
  //                 import(
  //                   "@/views/MoreMenu/MenuOne/MenuTwo/MenuThree/index-2.vue"
  //                 ),
  //             },
  //           ],
  //         },
  //       ],
  //     },
  //   ],
  // },
  {
    path: "/404",
    meta: { title: "404", isShow: false },
    component: () => import("@/views/404/index.vue"),
  },
]
/****
 * @异步路由也叫权限路由
 */
export const asyncRouter = [
  
]
/***
 * @任意路由
 */
export const anyRouter = [
  //任意路由
  {
    path: "/:pathMatch(.*)*",
    // name: 'Any',
    meta: { title: "任意路由", isShow: false },
    redirect: "/404",
  },
]

// export const getHomeRouteConfig = () => {
//   const layoutRoute = constRouter.find(route => route.path === "/")
//   return layoutRoute?.children?.find(child => child.path === "/home")
// }

export const getLayoutChildrenRoutes = () => {
  const platformParent = constRouter.find(route => route.path === "/");
  const learnerinfo = constRouter.find(route => route.path === "/learnerinfo");
  const homeParent = constRouter.find(route => route.path === "/home");
  
  return [
    platformParent?.children?.find(child => child.path === "/platformstats"),
    learnerinfo?.children?.find(child => child.path === "/learnerinfo"),
    homeParent?.children?.find(child => child.path === "/home")
  ].filter(Boolean); // 过滤掉可能的undefined
};
