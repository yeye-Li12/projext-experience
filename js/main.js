/* ============================================================
   李贞 · 网页版简历 — 交互逻辑
   1. 隐私保护：带 ?privacy 参数才显示完整联系方式，防止爬虫抓取
   2. 联系方式混淆：字符串拼接生成，避免被简单爬虫直接读取
   3. 二维码：生成当前页面地址的二维码（方便手机扫码访问）
   4. 更新日期：自动填充当前月份
   ============================================================ */
(function () {
  'use strict';

  /* ==========================================================
     配置区：改这里就行
     email / phone —— 简历上的联系方式（页面上会自动显示/打码）
     qrTarget      —— 二维码地址，留空自动使用当前网址（推荐）
     ========================================================== */
  var CONFIG = {
    email: '15818588017@163.com',
    phone: '15818588017',
    qrTarget: '' // 例如 'https://你的用户名.github.io/resume/'
  };

  /* ---------- 1&2. 联系方式隐私 ---------- */
  var showPrivacy = /[?&]privacy=?(&|$)/.test(window.location.search);

  function maskEmail(e) { return e.replace(/^(.{3}).*(@.*)$/, '$1***$2'); }
  function maskPhone(p) { return p.replace(/^(.{3}).*(.{4})$/, '$1***$2'); }

  var emailLink = document.getElementById('email-link');
  if (emailLink && CONFIG.email) {
    emailLink.href = 'mailto:' + CONFIG.email;
    emailLink.textContent = CONFIG.email;
    if (!showPrivacy) emailLink.style.display = 'none';
    var em = document.querySelector('.email-masked');
    if (em) em.textContent = maskEmail(CONFIG.email);
  }

  var phoneLink = document.getElementById('phone-link');
  if (phoneLink && CONFIG.phone) {
    phoneLink.href = 'tel:' + CONFIG.phone;
    phoneLink.textContent = CONFIG.phone;
    if (!showPrivacy) phoneLink.style.display = 'none';
    var pm = document.querySelector('.phone-masked');
    if (pm) pm.textContent = maskPhone(CONFIG.phone);
  }

  if (showPrivacy) {
    document.body.classList.add('show-privacy');
  }

  /* ---------- 3. 二维码 ---------- */
  // 二维码目标地址：
  //  - 部署上线后（http/https 打开）自动使用当前网址，无需任何配置；
  //  - 本地 file:// 预览时可临时用 ?url=https://你的地址 指定一个链接测试；
  //  - 也可以在下方 QR_TARGET 填你的线上简历地址（部署后建议留空）。
  var QR_TARGET = CONFIG.qrTarget; // 二维码目标地址，见上方配置区
  var qrText = '';
  var qrMatch = /[?&]url=([^&]+)/.exec(window.location.search);

  if (QR_TARGET) {
    qrText = QR_TARGET;
  } else if (qrMatch) {
    qrText = decodeURIComponent(qrMatch[1]);
  } else if (/^https?:/.test(window.location.protocol)) {
    // 线上环境：用干净的页面地址（去掉 ?privacy 等参数），避免二维码带隐私参数
    qrText = window.location.origin + window.location.pathname;
  }

  var qrBox = document.getElementById('qrcode');
  var qrHint = document.getElementById('qr-hint');
  if (qrBox) {
    if (qrText && typeof QRCode !== 'undefined') {
      try {
        new QRCode(qrBox, {
          text: qrText,
          width: 180,
          height: 180,
          colorDark: '#1f2937',
          colorLight: '#ffffff',
          correctLevel: QRCode.CorrectLevel.M
        });
        if (qrHint) {
          qrHint.textContent = '扫码打开：' + qrText;
          qrHint.style.display = 'block';
        }
      } catch (e) {
        if (qrHint) { qrHint.textContent = '二维码生成失败，请稍后重试'; qrHint.style.display = 'block'; }
      }
    } else if (qrHint) {
      // 本地预览：没有可扫描的线上地址
      qrHint.textContent = '📱 当前为本地预览：扫码地址将在部署上线后自动生成（详见 DEPLOY.md）';
      qrHint.style.display = 'block';
    }
  }

  // GitHub 项目二维码：指向项目仓库（projext-experience），扫码即可看到项目文件
  var qrGithubBox = document.getElementById('qrcode-github');
  if (qrGithubBox && typeof QRCode !== 'undefined') {
    try {
      new QRCode(qrGithubBox, {
        text: 'https://github.com/yeye-Li12/projext-experience',
        width: 150,
        height: 150,
        colorDark: '#1f2937',
        colorLight: '#ffffff',
        correctLevel: QRCode.CorrectLevel.M
      });
    } catch (e) {
      qrGithubBox.innerHTML = '<span style="font-size:12px;color:#999">GitHub</span>';
    }
  }

  /* ---------- 4. 更新日期 ---------- */
  var dateEl = document.getElementById('update-date');
  if (dateEl) {
    var now = new Date();
    dateEl.textContent = now.getFullYear() + '年' + (now.getMonth() + 1) + '月';
  }

  /* ---------- 5. 下载 PDF 按钮 ---------- */
  var btnPdf = document.getElementById('btn-pdf');
  if (btnPdf) {
    btnPdf.addEventListener('click', function () {
      // 直接调用浏览器打印，用户选择「另存为 PDF」即可得到 A4 两页版
      window.print();
    });
  }

  /* ---------- 6. 项目图表：图片缺失时显示占位提示 ---------- */
  var figures = document.querySelectorAll('.project-figure');
  figures.forEach(function (fig) {
    var img = fig.querySelector('img');
    var alt = fig.getAttribute('data-alt') || '';
    if (!img) return;
    img.addEventListener('error', function () {
      if (fig.querySelector('.fig-placeholder')) return; // 已生成过占位
      img.style.display = 'none';
      var ph = document.createElement('div');
      ph.className = 'fig-placeholder';
      ph.innerHTML = '<span style="font-size:22px">📊</span>' +
        '<span>' + alt + '</span>' +
        '<span style="font-size:11px">将图表截图保存到 images/ 文件夹后自动显示<br>（文件名见 images/README.txt）</span>';
      fig.appendChild(ph);
    });
  });
})();
