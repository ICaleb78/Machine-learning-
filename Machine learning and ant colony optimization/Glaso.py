<!DOCTYPE html>
<!-- saved from url=(0104)file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb.html -->
<html lang="en-US" style="--jp-side-by-side-output-size: 1fr;"><head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8"><style>.ͼ1.cm-focused {outline: 1px dotted #212121;}
.ͼ1 {position: relative !important; box-sizing: border-box; display: flex !important; flex-direction: column;}
.ͼ1 .cm-scroller {display: flex !important; align-items: flex-start !important; font-family: monospace; line-height: 1.4; height: 100%; overflow-x: auto; position: relative; z-index: 0;}
.ͼ1 .cm-content[contenteditable=true] {-webkit-user-modify: read-write-plaintext-only;}
.ͼ1 .cm-content {margin: 0; flex-grow: 2; flex-shrink: 0; display: block; white-space: pre; word-wrap: normal; box-sizing: border-box; padding: 4px 0; outline: none;}
.ͼ1 .cm-lineWrapping {white-space: pre-wrap; white-space: break-spaces; word-break: break-word; overflow-wrap: anywhere; flex-shrink: 1;}
.ͼ2 .cm-content {caret-color: black;}
.ͼ3 .cm-content {caret-color: white;}
.ͼ1 .cm-line {display: block; padding: 0 2px 0 6px;}
.ͼ1 .cm-layer > * {position: absolute;}
.ͼ1 .cm-layer {position: absolute; left: 0; top: 0; contain: size style;}
.ͼ2 .cm-selectionBackground {background: #d9d9d9;}
.ͼ3 .cm-selectionBackground {background: #222;}
.ͼ2.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground {background: #d7d4f0;}
.ͼ3.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground {background: #233;}
.ͼ1 .cm-cursorLayer {pointer-events: none;}
.ͼ1.cm-focused > .cm-scroller > .cm-cursorLayer {animation: steps(1) cm-blink 1.2s infinite;}
@keyframes cm-blink {50% {opacity: 0;}}
@keyframes cm-blink2 {50% {opacity: 0;}}
.ͼ1 .cm-cursor, .ͼ1 .cm-dropCursor {border-left: 1.2px solid black; margin-left: -0.6px; pointer-events: none;}
.ͼ1 .cm-cursor {display: none;}
.ͼ3 .cm-cursor {border-left-color: #444;}
.ͼ1 .cm-dropCursor {position: absolute;}
.ͼ1.cm-focused > .cm-scroller > .cm-cursorLayer .cm-cursor {display: block;}
.ͼ2 .cm-activeLine {background-color: #cceeff44;}
.ͼ3 .cm-activeLine {background-color: #99eeff33;}
.ͼ2 .cm-specialChar {color: red;}
.ͼ3 .cm-specialChar {color: #f78;}
.ͼ1 .cm-gutters {flex-shrink: 0; display: flex; height: 100%; box-sizing: border-box; left: 0; z-index: 200;}
.ͼ2 .cm-gutters {background-color: #f5f5f5; color: #6c6c6c; border-right: 1px solid #ddd;}
.ͼ3 .cm-gutters {background-color: #333338; color: #ccc;}
.ͼ1 .cm-gutter {display: flex !important; flex-direction: column; flex-shrink: 0; box-sizing: border-box; min-height: 100%; overflow: hidden;}
.ͼ1 .cm-gutterElement {box-sizing: border-box;}
.ͼ1 .cm-lineNumbers .cm-gutterElement {padding: 0 3px 0 5px; min-width: 20px; text-align: right; white-space: nowrap;}
.ͼ2 .cm-activeLineGutter {background-color: #e2f2ff;}
.ͼ3 .cm-activeLineGutter {background-color: #222227;}
.ͼ1 .cm-panels {box-sizing: border-box; position: sticky; left: 0; right: 0;}
.ͼ2 .cm-panels {background-color: #f5f5f5; color: black;}
.ͼ2 .cm-panels-top {border-bottom: 1px solid #ddd;}
.ͼ2 .cm-panels-bottom {border-top: 1px solid #ddd;}
.ͼ3 .cm-panels {background-color: #333338; color: white;}
.ͼ1 .cm-tab {display: inline-block; overflow: hidden; vertical-align: bottom;}
.ͼ1 .cm-widgetBuffer {vertical-align: text-top; height: 1em; width: 0; display: inline;}
.ͼ1 .cm-placeholder {color: #888; display: inline-block; vertical-align: top;}
.ͼ1 .cm-highlightSpace:before {content: attr(data-display); position: absolute; pointer-events: none; color: #888;}
.ͼ1 .cm-highlightTab {background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="20"><path stroke="%23888" stroke-width="1" fill="none" d="M1 10H196L190 5M190 15L196 10M197 4L197 16"/></svg>'); background-size: auto 100%; background-position: right 90%; background-repeat: no-repeat;}
.ͼ1 .cm-trailingSpace {background-color: #ff332255;}
.ͼ1 .cm-button {vertical-align: middle; color: inherit; font-size: 70%; padding: .2em 1em; border-radius: 1px;}
.ͼ2 .cm-button:active {background-image: linear-gradient(#b4b4b4, #d0d3d6);}
.ͼ2 .cm-button {background-image: linear-gradient(#eff1f5, #d9d9df); border: 1px solid #888;}
.ͼ3 .cm-button:active {background-image: linear-gradient(#111, #333);}
.ͼ3 .cm-button {background-image: linear-gradient(#393939, #111); border: 1px solid #888;}
.ͼ1 .cm-textfield {vertical-align: middle; color: inherit; font-size: 70%; border: 1px solid silver; padding: .2em .5em;}
.ͼ2 .cm-textfield {background-color: white;}
.ͼ3 .cm-textfield {border: 1px solid #555; background-color: inherit;}
.ͼ1.cm-focused .cm-matchingBracket {background-color: #328c8252;}
.ͼ1.cm-focused .cm-nonmatchingBracket {background-color: #bb555544;}
.ͼp {color: var(--jp-mirror-editor-meta-color);}
.ͼq {color: var(--jp-mirror-editor-header-color);}
.ͼr {color: var(--jp-mirror-editor-header-color); font-weight: bold;}
.ͼs {color: var(--jp-mirror-editor-keyword-color); font-weight: bold;}
.ͼt {color: var(--jp-mirror-editor-atom-color);}
.ͼu {color: var(--jp-mirror-editor-number-color);}
.ͼv {color: var(--jp-mirror-editor-def-color);}
.ͼw {color: var(--jp-mirror-editor-builtin-color);}
.ͼx {color: var(--jp-mirror-editor-variable-2-color);}
.ͼy {color: var(--jp-mirror-editor-punctuation-color);}
.ͼz {color: var(--jp-mirror-editor-property-color);}
.ͼ10 {color: var(--jp-mirror-editor-operator-color); font-weight: bold;}
.ͼ11 {color: var(--jp-mirror-editor-comment-color); font-style: italic;}
.ͼ12 {color: var(--jp-mirror-editor-string-color);}
.ͼ13 {color: var(--jp-mirror-editor-string-2-color);}
.ͼ14 {color: var(--jp-mirror-editor-bracket-color);}
.ͼ15 {color: var(--jp-mirror-editor-tag-color);}
.ͼ16 {color: var(--jp-mirror-editor-attribute-color);}
.ͼ17 {color: var(--jp-mirror-editor-quote-color);}
.ͼ18 {color: var(--jp-mirror-editor-link-color); text-decoration: underline;}
.ͼ19 {color: ;}
.ͼ1a {font-weight: bold;}
.ͼ1b {font-style: italic;}
.ͼ1c {text-decoration: line-through;}
.ͼ1d {color: var(--jp-mirror-editor-keyword-color); font-weight: bold;}
.ͼo {background: var(--jp-layout-color0); color: var(--jp-content-font-color1);}
.jp-CodeConsole .ͼo, .jp-Notebook .ͼo {background: transparent;}
.ͼo .cm-content {caret-color: var(--jp-editor-cursor-color);}
.ͼo .cm-scroller {font-family: inherit;}
.ͼo .cm-cursor, .ͼo .cm-dropCursor {border-left: var(--jp-code-cursor-width0) solid var(--jp-editor-cursor-color);}
.ͼo .cm-selectionBackground, .ͼo .cm-content ::selection {background-color: var(--jp-editor-selected-background);}
.ͼo.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground {background-color: var(--jp-editor-selected-focused-background);}
.ͼo .cm-gutters {border-right: 1px solid var(--jp-border-color2); background-color: var(--jp-layout-color2);}
.ͼo .cm-gutter {background-color: var(--jp-layout-color2);}
.ͼo .cm-activeLine {background-color: color-mix(in srgb, var(--jp-layout-color3) 25%, transparent);}
.ͼo .cm-lineNumbers {color: var(--jp-ui-font-color2);}
.ͼo .cm-searchMatch {background-color: var(--jp-search-unselected-match-background-color); color: var(--jp-search-unselected-match-color);}
.ͼo .cm-searchMatch.cm-searchMatch-selected {background-color: var(--jp-search-selected-match-background-color) !important; color: var(--jp-search-selected-match-color) !important;}
.ͼo .cm-tooltip {background-color: var(--jp-layout-color1);}
.ͼ4 .cm-line ::selection {background-color: transparent !important;}
.ͼ4 .cm-line::selection {background-color: transparent !important;}
.ͼ4 .cm-line {caret-color: transparent !important;}
</style>
  
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Glaso</title>
  
  <link rel="icon" type="image/x-icon" href="http://localhost:8888/static/favicons/favicon-notebook.ico" class="favicon">
  

  
  <link rel="stylesheet" type="text/css" href="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/custom.css">
  
<style>/*!
 * Font Awesome Free 5.15.4 by @fontawesome - https://fontawesome.com
 * License - https://fontawesome.com/license/free (Icons: CC BY 4.0, Fonts: SIL OFL 1.1, Code: MIT License)
 */
.fa,.fab,.fad,.fal,.far,.fas{-moz-osx-font-smoothing:grayscale;-webkit-font-smoothing:antialiased;display:inline-block;font-style:normal;font-variant:normal;text-rendering:auto;line-height:1}.fa-lg{font-size:1.33333em;line-height:.75em;vertical-align:-.0667em}.fa-xs{font-size:.75em}.fa-sm{font-size:.875em}.fa-1x{font-size:1em}.fa-2x{font-size:2em}.fa-3x{font-size:3em}.fa-4x{font-size:4em}.fa-5x{font-size:5em}.fa-6x{font-size:6em}.fa-7x{font-size:7em}.fa-8x{font-size:8em}.fa-9x{font-size:9em}.fa-10x{font-size:10em}.fa-fw{text-align:center;width:1.25em}.fa-ul{list-style-type:none;margin-left:2.5em;padding-left:0}.fa-ul>li{position:relative}.fa-li{left:-2em;position:absolute;text-align:center;width:2em;line-height:inherit}.fa-border{border:.08em solid #eee;border-radius:.1em;padding:.2em .25em .15em}.fa-pull-left{float:left}.fa-pull-right{float:right}.fa.fa-pull-left,.fab.fa-pull-left,.fal.fa-pull-left,.far.fa-pull-left,.fas.fa-pull-left{margin-right:.3em}.fa.fa-pull-right,.fab.fa-pull-right,.fal.fa-pull-right,.far.fa-pull-right,.fas.fa-pull-right{margin-left:.3em}.fa-spin{-webkit-animation:fa-spin 2s linear infinite;animation:fa-spin 2s linear infinite}.fa-pulse{-webkit-animation:fa-spin 1s steps(8) infinite;animation:fa-spin 1s steps(8) infinite}@-webkit-keyframes fa-spin{0%{-webkit-transform:rotate(0deg);transform:rotate(0deg)}to{-webkit-transform:rotate(1turn);transform:rotate(1turn)}}@keyframes fa-spin{0%{-webkit-transform:rotate(0deg);transform:rotate(0deg)}to{-webkit-transform:rotate(1turn);transform:rotate(1turn)}}.fa-rotate-90{-ms-filter:"progid:DXImageTransform.Microsoft.BasicImage(rotation=1)";-webkit-transform:rotate(90deg);transform:rotate(90deg)}.fa-rotate-180{-ms-filter:"progid:DXImageTransform.Microsoft.BasicImage(rotation=2)";-webkit-transform:rotate(180deg);transform:rotate(180deg)}.fa-rotate-270{-ms-filter:"progid:DXImageTransform.Microsoft.BasicImage(rotation=3)";-webkit-transform:rotate(270deg);transform:rotate(270deg)}.fa-flip-horizontal{-ms-filter:"progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1)";-webkit-transform:scaleX(-1);transform:scaleX(-1)}.fa-flip-vertical{-webkit-transform:scaleY(-1);transform:scaleY(-1)}.fa-flip-both,.fa-flip-horizontal.fa-flip-vertical,.fa-flip-vertical{-ms-filter:"progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1)"}.fa-flip-both,.fa-flip-horizontal.fa-flip-vertical{-webkit-transform:scale(-1);transform:scale(-1)}:root .fa-flip-both,:root .fa-flip-horizontal,:root .fa-flip-vertical,:root .fa-rotate-90,:root .fa-rotate-180,:root .fa-rotate-270{-webkit-filter:none;filter:none}.fa-stack{display:inline-block;height:2em;line-height:2em;position:relative;vertical-align:middle;width:2.5em}.fa-stack-1x,.fa-stack-2x{left:0;position:absolute;text-align:center;width:100%}.fa-stack-1x{line-height:inherit}.fa-stack-2x{font-size:2em}.fa-inverse{color:#fff}.fa-500px:before{content:"\f26e"}.fa-accessible-icon:before{content:"\f368"}.fa-accusoft:before{content:"\f369"}.fa-acquisitions-incorporated:before{content:"\f6af"}.fa-ad:before{content:"\f641"}.fa-address-book:before{content:"\f2b9"}.fa-address-card:before{content:"\f2bb"}.fa-adjust:before{content:"\f042"}.fa-adn:before{content:"\f170"}.fa-adversal:before{content:"\f36a"}.fa-affiliatetheme:before{content:"\f36b"}.fa-air-freshener:before{content:"\f5d0"}.fa-airbnb:before{content:"\f834"}.fa-algolia:before{content:"\f36c"}.fa-align-center:before{content:"\f037"}.fa-align-justify:before{content:"\f039"}.fa-align-left:before{content:"\f036"}.fa-align-right:before{content:"\f038"}.fa-alipay:before{content:"\f642"}.fa-allergies:before{content:"\f461"}.fa-amazon:before{content:"\f270"}.fa-amazon-pay:before{content:"\f42c"}.fa-ambulance:before{content:"\f0f9"}.fa-american-sign-language-interpreting:before{content:"\f2a3"}.fa-amilia:before{content:"\f36d"}.fa-anchor:before{content:"\f13d"}.fa-android:before{content:"\f17b"}.fa-angellist:before{content:"\f209"}.fa-angle-double-down:before{content:"\f103"}.fa-angle-double-left:before{content:"\f100"}.fa-angle-double-right:before{content:"\f101"}.fa-angle-double-up:before{content:"\f102"}.fa-angle-down:before{content:"\f107"}.fa-angle-left:before{content:"\f104"}.fa-angle-right:before{content:"\f105"}.fa-angle-up:before{content:"\f106"}.fa-angry:before{content:"\f556"}.fa-angrycreative:before{content:"\f36e"}.fa-angular:before{content:"\f420"}.fa-ankh:before{content:"\f644"}.fa-app-store:before{content:"\f36f"}.fa-app-store-ios:before{content:"\f370"}.fa-apper:before{content:"\f371"}.fa-apple:before{content:"\f179"}.fa-apple-alt:before{content:"\f5d1"}.fa-apple-pay:before{content:"\f415"}.fa-archive:before{content:"\f187"}.fa-archway:before{content:"\f557"}.fa-arrow-alt-circle-down:before{content:"\f358"}.fa-arrow-alt-circle-left:before{content:"\f359"}.fa-arrow-alt-circle-right:before{content:"\f35a"}.fa-arrow-alt-circle-up:before{content:"\f35b"}.fa-arrow-circle-down:before{content:"\f0ab"}.fa-arrow-circle-left:before{content:"\f0a8"}.fa-arrow-circle-right:before{content:"\f0a9"}.fa-arrow-circle-up:before{content:"\f0aa"}.fa-arrow-down:before{content:"\f063"}.fa-arrow-left:before{content:"\f060"}.fa-arrow-right:before{content:"\f061"}.fa-arrow-up:before{content:"\f062"}.fa-arrows-alt:before{content:"\f0b2"}.fa-arrows-alt-h:before{content:"\f337"}.fa-arrows-alt-v:before{content:"\f338"}.fa-artstation:before{content:"\f77a"}.fa-assistive-listening-systems:before{content:"\f2a2"}.fa-asterisk:before{content:"\f069"}.fa-asymmetrik:before{content:"\f372"}.fa-at:before{content:"\f1fa"}.fa-atlas:before{content:"\f558"}.fa-atlassian:before{content:"\f77b"}.fa-atom:before{content:"\f5d2"}.fa-audible:before{content:"\f373"}.fa-audio-description:before{content:"\f29e"}.fa-autoprefixer:before{content:"\f41c"}.fa-avianex:before{content:"\f374"}.fa-aviato:before{content:"\f421"}.fa-award:before{content:"\f559"}.fa-aws:before{content:"\f375"}.fa-baby:before{content:"\f77c"}.fa-baby-carriage:before{content:"\f77d"}.fa-backspace:before{content:"\f55a"}.fa-backward:before{content:"\f04a"}.fa-bacon:before{content:"\f7e5"}.fa-bacteria:before{content:"\e059"}.fa-bacterium:before{content:"\e05a"}.fa-bahai:before{content:"\f666"}.fa-balance-scale:before{content:"\f24e"}.fa-balance-scale-left:before{content:"\f515"}.fa-balance-scale-right:before{content:"\f516"}.fa-ban:before{content:"\f05e"}.fa-band-aid:before{content:"\f462"}.fa-bandcamp:before{content:"\f2d5"}.fa-barcode:before{content:"\f02a"}.fa-bars:before{content:"\f0c9"}.fa-baseball-ball:before{content:"\f433"}.fa-basketball-ball:before{content:"\f434"}.fa-bath:before{content:"\f2cd"}.fa-battery-empty:before{content:"\f244"}.fa-battery-full:before{content:"\f240"}.fa-battery-half:before{content:"\f242"}.fa-battery-quarter:before{content:"\f243"}.fa-battery-three-quarters:before{content:"\f241"}.fa-battle-net:before{content:"\f835"}.fa-bed:before{content:"\f236"}.fa-beer:before{content:"\f0fc"}.fa-behance:before{content:"\f1b4"}.fa-behance-square:before{content:"\f1b5"}.fa-bell:before{content:"\f0f3"}.fa-bell-slash:before{content:"\f1f6"}.fa-bezier-curve:before{content:"\f55b"}.fa-bible:before{content:"\f647"}.fa-bicycle:before{content:"\f206"}.fa-biking:before{content:"\f84a"}.fa-bimobject:before{content:"\f378"}.fa-binoculars:before{content:"\f1e5"}.fa-biohazard:before{content:"\f780"}.fa-birthday-cake:before{content:"\f1fd"}.fa-bitbucket:before{content:"\f171"}.fa-bitcoin:before{content:"\f379"}.fa-bity:before{content:"\f37a"}.fa-black-tie:before{content:"\f27e"}.fa-blackberry:before{content:"\f37b"}.fa-blender:before{content:"\f517"}.fa-blender-phone:before{content:"\f6b6"}.fa-blind:before{content:"\f29d"}.fa-blog:before{content:"\f781"}.fa-blogger:before{content:"\f37c"}.fa-blogger-b:before{content:"\f37d"}.fa-bluetooth:before{content:"\f293"}.fa-bluetooth-b:before{content:"\f294"}.fa-bold:before{content:"\f032"}.fa-bolt:before{content:"\f0e7"}.fa-bomb:before{content:"\f1e2"}.fa-bone:before{content:"\f5d7"}.fa-bong:before{content:"\f55c"}.fa-book:before{content:"\f02d"}.fa-book-dead:before{content:"\f6b7"}.fa-book-medical:before{content:"\f7e6"}.fa-book-open:before{content:"\f518"}.fa-book-reader:before{content:"\f5da"}.fa-bookmark:before{content:"\f02e"}.fa-bootstrap:before{content:"\f836"}.fa-border-all:before{content:"\f84c"}.fa-border-none:before{content:"\f850"}.fa-border-style:before{content:"\f853"}.fa-bowling-ball:before{content:"\f436"}.fa-box:before{content:"\f466"}.fa-box-open:before{content:"\f49e"}.fa-box-tissue:before{content:"\e05b"}.fa-boxes:before{content:"\f468"}.fa-braille:before{content:"\f2a1"}.fa-brain:before{content:"\f5dc"}.fa-bread-slice:before{content:"\f7ec"}.fa-briefcase:before{content:"\f0b1"}.fa-briefcase-medical:before{content:"\f469"}.fa-broadcast-tower:before{content:"\f519"}.fa-broom:before{content:"\f51a"}.fa-brush:before{content:"\f55d"}.fa-btc:before{content:"\f15a"}.fa-buffer:before{content:"\f837"}.fa-bug:before{content:"\f188"}.fa-building:before{content:"\f1ad"}.fa-bullhorn:before{content:"\f0a1"}.fa-bullseye:before{content:"\f140"}.fa-burn:before{content:"\f46a"}.fa-buromobelexperte:before{content:"\f37f"}.fa-bus:before{content:"\f207"}.fa-bus-alt:before{content:"\f55e"}.fa-business-time:before{content:"\f64a"}.fa-buy-n-large:before{content:"\f8a6"}.fa-buysellads:before{content:"\f20d"}.fa-calculator:before{content:"\f1ec"}.fa-calendar:before{content:"\f133"}.fa-calendar-alt:before{content:"\f073"}.fa-calendar-check:before{content:"\f274"}.fa-calendar-day:before{content:"\f783"}.fa-calendar-minus:before{content:"\f272"}.fa-calendar-plus:before{content:"\f271"}.fa-calendar-times:before{content:"\f273"}.fa-calendar-week:before{content:"\f784"}.fa-camera:before{content:"\f030"}.fa-camera-retro:before{content:"\f083"}.fa-campground:before{content:"\f6bb"}.fa-canadian-maple-leaf:before{content:"\f785"}.fa-candy-cane:before{content:"\f786"}.fa-cannabis:before{content:"\f55f"}.fa-capsules:before{content:"\f46b"}.fa-car:before{content:"\f1b9"}.fa-car-alt:before{content:"\f5de"}.fa-car-battery:before{content:"\f5df"}.fa-car-crash:before{content:"\f5e1"}.fa-car-side:before{content:"\f5e4"}.fa-caravan:before{content:"\f8ff"}.fa-caret-down:before{content:"\f0d7"}.fa-caret-left:before{content:"\f0d9"}.fa-caret-right:before{content:"\f0da"}.fa-caret-square-down:before{content:"\f150"}.fa-caret-square-left:before{content:"\f191"}.fa-caret-square-right:before{content:"\f152"}.fa-caret-square-up:before{content:"\f151"}.fa-caret-up:before{content:"\f0d8"}.fa-carrot:before{content:"\f787"}.fa-cart-arrow-down:before{content:"\f218"}.fa-cart-plus:before{content:"\f217"}.fa-cash-register:before{content:"\f788"}.fa-cat:before{content:"\f6be"}.fa-cc-amazon-pay:before{content:"\f42d"}.fa-cc-amex:before{content:"\f1f3"}.fa-cc-apple-pay:before{content:"\f416"}.fa-cc-diners-club:before{content:"\f24c"}.fa-cc-discover:before{content:"\f1f2"}.fa-cc-jcb:before{content:"\f24b"}.fa-cc-mastercard:before{content:"\f1f1"}.fa-cc-paypal:before{content:"\f1f4"}.fa-cc-stripe:before{content:"\f1f5"}.fa-cc-visa:before{content:"\f1f0"}.fa-centercode:before{content:"\f380"}.fa-centos:before{content:"\f789"}.fa-certificate:before{content:"\f0a3"}.fa-chair:before{content:"\f6c0"}.fa-chalkboard:before{content:"\f51b"}.fa-chalkboard-teacher:before{content:"\f51c"}.fa-charging-station:before{content:"\f5e7"}.fa-chart-area:before{content:"\f1fe"}.fa-chart-bar:before{content:"\f080"}.fa-chart-line:before{content:"\f201"}.fa-chart-pie:before{content:"\f200"}.fa-check:before{content:"\f00c"}.fa-check-circle:before{content:"\f058"}.fa-check-double:before{content:"\f560"}.fa-check-square:before{content:"\f14a"}.fa-cheese:before{content:"\f7ef"}.fa-chess:before{content:"\f439"}.fa-chess-bishop:before{content:"\f43a"}.fa-chess-board:before{content:"\f43c"}.fa-chess-king:before{content:"\f43f"}.fa-chess-knight:before{content:"\f441"}.fa-chess-pawn:before{content:"\f443"}.fa-chess-queen:before{content:"\f445"}.fa-chess-rook:before{content:"\f447"}.fa-chevron-circle-down:before{content:"\f13a"}.fa-chevron-circle-left:before{content:"\f137"}.fa-chevron-circle-right:before{content:"\f138"}.fa-chevron-circle-up:before{content:"\f139"}.fa-chevron-down:before{content:"\f078"}.fa-chevron-left:before{content:"\f053"}.fa-chevron-right:before{content:"\f054"}.fa-chevron-up:before{content:"\f077"}.fa-child:before{content:"\f1ae"}.fa-chrome:before{content:"\f268"}.fa-chromecast:before{content:"\f838"}.fa-church:before{content:"\f51d"}.fa-circle:before{content:"\f111"}.fa-circle-notch:before{content:"\f1ce"}.fa-city:before{content:"\f64f"}.fa-clinic-medical:before{content:"\f7f2"}.fa-clipboard:before{content:"\f328"}.fa-clipboard-check:before{content:"\f46c"}.fa-clipboard-list:before{content:"\f46d"}.fa-clock:before{content:"\f017"}.fa-clone:before{content:"\f24d"}.fa-closed-captioning:before{content:"\f20a"}.fa-cloud:before{content:"\f0c2"}.fa-cloud-download-alt:before{content:"\f381"}.fa-cloud-meatball:before{content:"\f73b"}.fa-cloud-moon:before{content:"\f6c3"}.fa-cloud-moon-rain:before{content:"\f73c"}.fa-cloud-rain:before{content:"\f73d"}.fa-cloud-showers-heavy:before{content:"\f740"}.fa-cloud-sun:before{content:"\f6c4"}.fa-cloud-sun-rain:before{content:"\f743"}.fa-cloud-upload-alt:before{content:"\f382"}.fa-cloudflare:before{content:"\e07d"}.fa-cloudscale:before{content:"\f383"}.fa-cloudsmith:before{content:"\f384"}.fa-cloudversify:before{content:"\f385"}.fa-cocktail:before{content:"\f561"}.fa-code:before{content:"\f121"}.fa-code-branch:before{content:"\f126"}.fa-codepen:before{content:"\f1cb"}.fa-codiepie:before{content:"\f284"}.fa-coffee:before{content:"\f0f4"}.fa-cog:before{content:"\f013"}.fa-cogs:before{content:"\f085"}.fa-coins:before{content:"\f51e"}.fa-columns:before{content:"\f0db"}.fa-comment:before{content:"\f075"}.fa-comment-alt:before{content:"\f27a"}.fa-comment-dollar:before{content:"\f651"}.fa-comment-dots:before{content:"\f4ad"}.fa-comment-medical:before{content:"\f7f5"}.fa-comment-slash:before{content:"\f4b3"}.fa-comments:before{content:"\f086"}.fa-comments-dollar:before{content:"\f653"}.fa-compact-disc:before{content:"\f51f"}.fa-compass:before{content:"\f14e"}.fa-compress:before{content:"\f066"}.fa-compress-alt:before{content:"\f422"}.fa-compress-arrows-alt:before{content:"\f78c"}.fa-concierge-bell:before{content:"\f562"}.fa-confluence:before{content:"\f78d"}.fa-connectdevelop:before{content:"\f20e"}.fa-contao:before{content:"\f26d"}.fa-cookie:before{content:"\f563"}.fa-cookie-bite:before{content:"\f564"}.fa-copy:before{content:"\f0c5"}.fa-copyright:before{content:"\f1f9"}.fa-cotton-bureau:before{content:"\f89e"}.fa-couch:before{content:"\f4b8"}.fa-cpanel:before{content:"\f388"}.fa-creative-commons:before{content:"\f25e"}.fa-creative-commons-by:before{content:"\f4e7"}.fa-creative-commons-nc:before{content:"\f4e8"}.fa-creative-commons-nc-eu:before{content:"\f4e9"}.fa-creative-commons-nc-jp:before{content:"\f4ea"}.fa-creative-commons-nd:before{content:"\f4eb"}.fa-creative-commons-pd:before{content:"\f4ec"}.fa-creative-commons-pd-alt:before{content:"\f4ed"}.fa-creative-commons-remix:before{content:"\f4ee"}.fa-creative-commons-sa:before{content:"\f4ef"}.fa-creative-commons-sampling:before{content:"\f4f0"}.fa-creative-commons-sampling-plus:before{content:"\f4f1"}.fa-creative-commons-share:before{content:"\f4f2"}.fa-creative-commons-zero:before{content:"\f4f3"}.fa-credit-card:before{content:"\f09d"}.fa-critical-role:before{content:"\f6c9"}.fa-crop:before{content:"\f125"}.fa-crop-alt:before{content:"\f565"}.fa-cross:before{content:"\f654"}.fa-crosshairs:before{content:"\f05b"}.fa-crow:before{content:"\f520"}.fa-crown:before{content:"\f521"}.fa-crutch:before{content:"\f7f7"}.fa-css3:before{content:"\f13c"}.fa-css3-alt:before{content:"\f38b"}.fa-cube:before{content:"\f1b2"}.fa-cubes:before{content:"\f1b3"}.fa-cut:before{content:"\f0c4"}.fa-cuttlefish:before{content:"\f38c"}.fa-d-and-d:before{content:"\f38d"}.fa-d-and-d-beyond:before{content:"\f6ca"}.fa-dailymotion:before{content:"\e052"}.fa-dashcube:before{content:"\f210"}.fa-database:before{content:"\f1c0"}.fa-deaf:before{content:"\f2a4"}.fa-deezer:before{content:"\e077"}.fa-delicious:before{content:"\f1a5"}.fa-democrat:before{content:"\f747"}.fa-deploydog:before{content:"\f38e"}.fa-deskpro:before{content:"\f38f"}.fa-desktop:before{content:"\f108"}.fa-dev:before{content:"\f6cc"}.fa-deviantart:before{content:"\f1bd"}.fa-dharmachakra:before{content:"\f655"}.fa-dhl:before{content:"\f790"}.fa-diagnoses:before{content:"\f470"}.fa-diaspora:before{content:"\f791"}.fa-dice:before{content:"\f522"}.fa-dice-d20:before{content:"\f6cf"}.fa-dice-d6:before{content:"\f6d1"}.fa-dice-five:before{content:"\f523"}.fa-dice-four:before{content:"\f524"}.fa-dice-one:before{content:"\f525"}.fa-dice-six:before{content:"\f526"}.fa-dice-three:before{content:"\f527"}.fa-dice-two:before{content:"\f528"}.fa-digg:before{content:"\f1a6"}.fa-digital-ocean:before{content:"\f391"}.fa-digital-tachograph:before{content:"\f566"}.fa-directions:before{content:"\f5eb"}.fa-discord:before{content:"\f392"}.fa-discourse:before{content:"\f393"}.fa-disease:before{content:"\f7fa"}.fa-divide:before{content:"\f529"}.fa-dizzy:before{content:"\f567"}.fa-dna:before{content:"\f471"}.fa-dochub:before{content:"\f394"}.fa-docker:before{content:"\f395"}.fa-dog:before{content:"\f6d3"}.fa-dollar-sign:before{content:"\f155"}.fa-dolly:before{content:"\f472"}.fa-dolly-flatbed:before{content:"\f474"}.fa-donate:before{content:"\f4b9"}.fa-door-closed:before{content:"\f52a"}.fa-door-open:before{content:"\f52b"}.fa-dot-circle:before{content:"\f192"}.fa-dove:before{content:"\f4ba"}.fa-download:before{content:"\f019"}.fa-draft2digital:before{content:"\f396"}.fa-drafting-compass:before{content:"\f568"}.fa-dragon:before{content:"\f6d5"}.fa-draw-polygon:before{content:"\f5ee"}.fa-dribbble:before{content:"\f17d"}.fa-dribbble-square:before{content:"\f397"}.fa-dropbox:before{content:"\f16b"}.fa-drum:before{content:"\f569"}.fa-drum-steelpan:before{content:"\f56a"}.fa-drumstick-bite:before{content:"\f6d7"}.fa-drupal:before{content:"\f1a9"}.fa-dumbbell:before{content:"\f44b"}.fa-dumpster:before{content:"\f793"}.fa-dumpster-fire:before{content:"\f794"}.fa-dungeon:before{content:"\f6d9"}.fa-dyalog:before{content:"\f399"}.fa-earlybirds:before{content:"\f39a"}.fa-ebay:before{content:"\f4f4"}.fa-edge:before{content:"\f282"}.fa-edge-legacy:before{content:"\e078"}.fa-edit:before{content:"\f044"}.fa-egg:before{content:"\f7fb"}.fa-eject:before{content:"\f052"}.fa-elementor:before{content:"\f430"}.fa-ellipsis-h:before{content:"\f141"}.fa-ellipsis-v:before{content:"\f142"}.fa-ello:before{content:"\f5f1"}.fa-ember:before{content:"\f423"}.fa-empire:before{content:"\f1d1"}.fa-envelope:before{content:"\f0e0"}.fa-envelope-open:before{content:"\f2b6"}.fa-envelope-open-text:before{content:"\f658"}.fa-envelope-square:before{content:"\f199"}.fa-envira:before{content:"\f299"}.fa-equals:before{content:"\f52c"}.fa-eraser:before{content:"\f12d"}.fa-erlang:before{content:"\f39d"}.fa-ethereum:before{content:"\f42e"}.fa-ethernet:before{content:"\f796"}.fa-etsy:before{content:"\f2d7"}.fa-euro-sign:before{content:"\f153"}.fa-evernote:before{content:"\f839"}.fa-exchange-alt:before{content:"\f362"}.fa-exclamation:before{content:"\f12a"}.fa-exclamation-circle:before{content:"\f06a"}.fa-exclamation-triangle:before{content:"\f071"}.fa-expand:before{content:"\f065"}.fa-expand-alt:before{content:"\f424"}.fa-expand-arrows-alt:before{content:"\f31e"}.fa-expeditedssl:before{content:"\f23e"}.fa-external-link-alt:before{content:"\f35d"}.fa-external-link-square-alt:before{content:"\f360"}.fa-eye:before{content:"\f06e"}.fa-eye-dropper:before{content:"\f1fb"}.fa-eye-slash:before{content:"\f070"}.fa-facebook:before{content:"\f09a"}.fa-facebook-f:before{content:"\f39e"}.fa-facebook-messenger:before{content:"\f39f"}.fa-facebook-square:before{content:"\f082"}.fa-fan:before{content:"\f863"}.fa-fantasy-flight-games:before{content:"\f6dc"}.fa-fast-backward:before{content:"\f049"}.fa-fast-forward:before{content:"\f050"}.fa-faucet:before{content:"\e005"}.fa-fax:before{content:"\f1ac"}.fa-feather:before{content:"\f52d"}.fa-feather-alt:before{content:"\f56b"}.fa-fedex:before{content:"\f797"}.fa-fedora:before{content:"\f798"}.fa-female:before{content:"\f182"}.fa-fighter-jet:before{content:"\f0fb"}.fa-figma:before{content:"\f799"}.fa-file:before{content:"\f15b"}.fa-file-alt:before{content:"\f15c"}.fa-file-archive:before{content:"\f1c6"}.fa-file-audio:before{content:"\f1c7"}.fa-file-code:before{content:"\f1c9"}.fa-file-contract:before{content:"\f56c"}.fa-file-csv:before{content:"\f6dd"}.fa-file-download:before{content:"\f56d"}.fa-file-excel:before{content:"\f1c3"}.fa-file-export:before{content:"\f56e"}.fa-file-image:before{content:"\f1c5"}.fa-file-import:before{content:"\f56f"}.fa-file-invoice:before{content:"\f570"}.fa-file-invoice-dollar:before{content:"\f571"}.fa-file-medical:before{content:"\f477"}.fa-file-medical-alt:before{content:"\f478"}.fa-file-pdf:before{content:"\f1c1"}.fa-file-powerpoint:before{content:"\f1c4"}.fa-file-prescription:before{content:"\f572"}.fa-file-signature:before{content:"\f573"}.fa-file-upload:before{content:"\f574"}.fa-file-video:before{content:"\f1c8"}.fa-file-word:before{content:"\f1c2"}.fa-fill:before{content:"\f575"}.fa-fill-drip:before{content:"\f576"}.fa-film:before{content:"\f008"}.fa-filter:before{content:"\f0b0"}.fa-fingerprint:before{content:"\f577"}.fa-fire:before{content:"\f06d"}.fa-fire-alt:before{content:"\f7e4"}.fa-fire-extinguisher:before{content:"\f134"}.fa-firefox:before{content:"\f269"}.fa-firefox-browser:before{content:"\e007"}.fa-first-aid:before{content:"\f479"}.fa-first-order:before{content:"\f2b0"}.fa-first-order-alt:before{content:"\f50a"}.fa-firstdraft:before{content:"\f3a1"}.fa-fish:before{content:"\f578"}.fa-fist-raised:before{content:"\f6de"}.fa-flag:before{content:"\f024"}.fa-flag-checkered:before{content:"\f11e"}.fa-flag-usa:before{content:"\f74d"}.fa-flask:before{content:"\f0c3"}.fa-flickr:before{content:"\f16e"}.fa-flipboard:before{content:"\f44d"}.fa-flushed:before{content:"\f579"}.fa-fly:before{content:"\f417"}.fa-folder:before{content:"\f07b"}.fa-folder-minus:before{content:"\f65d"}.fa-folder-open:before{content:"\f07c"}.fa-folder-plus:before{content:"\f65e"}.fa-font:before{content:"\f031"}.fa-font-awesome:before{content:"\f2b4"}.fa-font-awesome-alt:before{content:"\f35c"}.fa-font-awesome-flag:before{content:"\f425"}.fa-font-awesome-logo-full:before{content:"\f4e6"}.fa-fonticons:before{content:"\f280"}.fa-fonticons-fi:before{content:"\f3a2"}.fa-football-ball:before{content:"\f44e"}.fa-fort-awesome:before{content:"\f286"}.fa-fort-awesome-alt:before{content:"\f3a3"}.fa-forumbee:before{content:"\f211"}.fa-forward:before{content:"\f04e"}.fa-foursquare:before{content:"\f180"}.fa-free-code-camp:before{content:"\f2c5"}.fa-freebsd:before{content:"\f3a4"}.fa-frog:before{content:"\f52e"}.fa-frown:before{content:"\f119"}.fa-frown-open:before{content:"\f57a"}.fa-fulcrum:before{content:"\f50b"}.fa-funnel-dollar:before{content:"\f662"}.fa-futbol:before{content:"\f1e3"}.fa-galactic-republic:before{content:"\f50c"}.fa-galactic-senate:before{content:"\f50d"}.fa-gamepad:before{content:"\f11b"}.fa-gas-pump:before{content:"\f52f"}.fa-gavel:before{content:"\f0e3"}.fa-gem:before{content:"\f3a5"}.fa-genderless:before{content:"\f22d"}.fa-get-pocket:before{content:"\f265"}.fa-gg:before{content:"\f260"}.fa-gg-circle:before{content:"\f261"}.fa-ghost:before{content:"\f6e2"}.fa-gift:before{content:"\f06b"}.fa-gifts:before{content:"\f79c"}.fa-git:before{content:"\f1d3"}.fa-git-alt:before{content:"\f841"}.fa-git-square:before{content:"\f1d2"}.fa-github:before{content:"\f09b"}.fa-github-alt:before{content:"\f113"}.fa-github-square:before{content:"\f092"}.fa-gitkraken:before{content:"\f3a6"}.fa-gitlab:before{content:"\f296"}.fa-gitter:before{content:"\f426"}.fa-glass-cheers:before{content:"\f79f"}.fa-glass-martini:before{content:"\f000"}.fa-glass-martini-alt:before{content:"\f57b"}.fa-glass-whiskey:before{content:"\f7a0"}.fa-glasses:before{content:"\f530"}.fa-glide:before{content:"\f2a5"}.fa-glide-g:before{content:"\f2a6"}.fa-globe:before{content:"\f0ac"}.fa-globe-africa:before{content:"\f57c"}.fa-globe-americas:before{content:"\f57d"}.fa-globe-asia:before{content:"\f57e"}.fa-globe-europe:before{content:"\f7a2"}.fa-gofore:before{content:"\f3a7"}.fa-golf-ball:before{content:"\f450"}.fa-goodreads:before{content:"\f3a8"}.fa-goodreads-g:before{content:"\f3a9"}.fa-google:before{content:"\f1a0"}.fa-google-drive:before{content:"\f3aa"}.fa-google-pay:before{content:"\e079"}.fa-google-play:before{content:"\f3ab"}.fa-google-plus:before{content:"\f2b3"}.fa-google-plus-g:before{content:"\f0d5"}.fa-google-plus-square:before{content:"\f0d4"}.fa-google-wallet:before{content:"\f1ee"}.fa-gopuram:before{content:"\f664"}.fa-graduation-cap:before{content:"\f19d"}.fa-gratipay:before{content:"\f184"}.fa-grav:before{content:"\f2d6"}.fa-greater-than:before{content:"\f531"}.fa-greater-than-equal:before{content:"\f532"}.fa-grimace:before{content:"\f57f"}.fa-grin:before{content:"\f580"}.fa-grin-alt:before{content:"\f581"}.fa-grin-beam:before{content:"\f582"}.fa-grin-beam-sweat:before{content:"\f583"}.fa-grin-hearts:before{content:"\f584"}.fa-grin-squint:before{content:"\f585"}.fa-grin-squint-tears:before{content:"\f586"}.fa-grin-stars:before{content:"\f587"}.fa-grin-tears:before{content:"\f588"}.fa-grin-tongue:before{content:"\f589"}.fa-grin-tongue-squint:before{content:"\f58a"}.fa-grin-tongue-wink:before{content:"\f58b"}.fa-grin-wink:before{content:"\f58c"}.fa-grip-horizontal:before{content:"\f58d"}.fa-grip-lines:before{content:"\f7a4"}.fa-grip-lines-vertical:before{content:"\f7a5"}.fa-grip-vertical:before{content:"\f58e"}.fa-gripfire:before{content:"\f3ac"}.fa-grunt:before{content:"\f3ad"}.fa-guilded:before{content:"\e07e"}.fa-guitar:before{content:"\f7a6"}.fa-gulp:before{content:"\f3ae"}.fa-h-square:before{content:"\f0fd"}.fa-hacker-news:before{content:"\f1d4"}.fa-hacker-news-square:before{content:"\f3af"}.fa-hackerrank:before{content:"\f5f7"}.fa-hamburger:before{content:"\f805"}.fa-hammer:before{content:"\f6e3"}.fa-hamsa:before{content:"\f665"}.fa-hand-holding:before{content:"\f4bd"}.fa-hand-holding-heart:before{content:"\f4be"}.fa-hand-holding-medical:before{content:"\e05c"}.fa-hand-holding-usd:before{content:"\f4c0"}.fa-hand-holding-water:before{content:"\f4c1"}.fa-hand-lizard:before{content:"\f258"}.fa-hand-middle-finger:before{content:"\f806"}.fa-hand-paper:before{content:"\f256"}.fa-hand-peace:before{content:"\f25b"}.fa-hand-point-down:before{content:"\f0a7"}.fa-hand-point-left:before{content:"\f0a5"}.fa-hand-point-right:before{content:"\f0a4"}.fa-hand-point-up:before{content:"\f0a6"}.fa-hand-pointer:before{content:"\f25a"}.fa-hand-rock:before{content:"\f255"}.fa-hand-scissors:before{content:"\f257"}.fa-hand-sparkles:before{content:"\e05d"}.fa-hand-spock:before{content:"\f259"}.fa-hands:before{content:"\f4c2"}.fa-hands-helping:before{content:"\f4c4"}.fa-hands-wash:before{content:"\e05e"}.fa-handshake:before{content:"\f2b5"}.fa-handshake-alt-slash:before{content:"\e05f"}.fa-handshake-slash:before{content:"\e060"}.fa-hanukiah:before{content:"\f6e6"}.fa-hard-hat:before{content:"\f807"}.fa-hashtag:before{content:"\f292"}.fa-hat-cowboy:before{content:"\f8c0"}.fa-hat-cowboy-side:before{content:"\f8c1"}.fa-hat-wizard:before{content:"\f6e8"}.fa-hdd:before{content:"\f0a0"}.fa-head-side-cough:before{content:"\e061"}.fa-head-side-cough-slash:before{content:"\e062"}.fa-head-side-mask:before{content:"\e063"}.fa-head-side-virus:before{content:"\e064"}.fa-heading:before{content:"\f1dc"}.fa-headphones:before{content:"\f025"}.fa-headphones-alt:before{content:"\f58f"}.fa-headset:before{content:"\f590"}.fa-heart:before{content:"\f004"}.fa-heart-broken:before{content:"\f7a9"}.fa-heartbeat:before{content:"\f21e"}.fa-helicopter:before{content:"\f533"}.fa-highlighter:before{content:"\f591"}.fa-hiking:before{content:"\f6ec"}.fa-hippo:before{content:"\f6ed"}.fa-hips:before{content:"\f452"}.fa-hire-a-helper:before{content:"\f3b0"}.fa-history:before{content:"\f1da"}.fa-hive:before{content:"\e07f"}.fa-hockey-puck:before{content:"\f453"}.fa-holly-berry:before{content:"\f7aa"}.fa-home:before{content:"\f015"}.fa-hooli:before{content:"\f427"}.fa-hornbill:before{content:"\f592"}.fa-horse:before{content:"\f6f0"}.fa-horse-head:before{content:"\f7ab"}.fa-hospital:before{content:"\f0f8"}.fa-hospital-alt:before{content:"\f47d"}.fa-hospital-symbol:before{content:"\f47e"}.fa-hospital-user:before{content:"\f80d"}.fa-hot-tub:before{content:"\f593"}.fa-hotdog:before{content:"\f80f"}.fa-hotel:before{content:"\f594"}.fa-hotjar:before{content:"\f3b1"}.fa-hourglass:before{content:"\f254"}.fa-hourglass-end:before{content:"\f253"}.fa-hourglass-half:before{content:"\f252"}.fa-hourglass-start:before{content:"\f251"}.fa-house-damage:before{content:"\f6f1"}.fa-house-user:before{content:"\e065"}.fa-houzz:before{content:"\f27c"}.fa-hryvnia:before{content:"\f6f2"}.fa-html5:before{content:"\f13b"}.fa-hubspot:before{content:"\f3b2"}.fa-i-cursor:before{content:"\f246"}.fa-ice-cream:before{content:"\f810"}.fa-icicles:before{content:"\f7ad"}.fa-icons:before{content:"\f86d"}.fa-id-badge:before{content:"\f2c1"}.fa-id-card:before{content:"\f2c2"}.fa-id-card-alt:before{content:"\f47f"}.fa-ideal:before{content:"\e013"}.fa-igloo:before{content:"\f7ae"}.fa-image:before{content:"\f03e"}.fa-images:before{content:"\f302"}.fa-imdb:before{content:"\f2d8"}.fa-inbox:before{content:"\f01c"}.fa-indent:before{content:"\f03c"}.fa-industry:before{content:"\f275"}.fa-infinity:before{content:"\f534"}.fa-info:before{content:"\f129"}.fa-info-circle:before{content:"\f05a"}.fa-innosoft:before{content:"\e080"}.fa-instagram:before{content:"\f16d"}.fa-instagram-square:before{content:"\e055"}.fa-instalod:before{content:"\e081"}.fa-intercom:before{content:"\f7af"}.fa-internet-explorer:before{content:"\f26b"}.fa-invision:before{content:"\f7b0"}.fa-ioxhost:before{content:"\f208"}.fa-italic:before{content:"\f033"}.fa-itch-io:before{content:"\f83a"}.fa-itunes:before{content:"\f3b4"}.fa-itunes-note:before{content:"\f3b5"}.fa-java:before{content:"\f4e4"}.fa-jedi:before{content:"\f669"}.fa-jedi-order:before{content:"\f50e"}.fa-jenkins:before{content:"\f3b6"}.fa-jira:before{content:"\f7b1"}.fa-joget:before{content:"\f3b7"}.fa-joint:before{content:"\f595"}.fa-joomla:before{content:"\f1aa"}.fa-journal-whills:before{content:"\f66a"}.fa-js:before{content:"\f3b8"}.fa-js-square:before{content:"\f3b9"}.fa-jsfiddle:before{content:"\f1cc"}.fa-kaaba:before{content:"\f66b"}.fa-kaggle:before{content:"\f5fa"}.fa-key:before{content:"\f084"}.fa-keybase:before{content:"\f4f5"}.fa-keyboard:before{content:"\f11c"}.fa-keycdn:before{content:"\f3ba"}.fa-khanda:before{content:"\f66d"}.fa-kickstarter:before{content:"\f3bb"}.fa-kickstarter-k:before{content:"\f3bc"}.fa-kiss:before{content:"\f596"}.fa-kiss-beam:before{content:"\f597"}.fa-kiss-wink-heart:before{content:"\f598"}.fa-kiwi-bird:before{content:"\f535"}.fa-korvue:before{content:"\f42f"}.fa-landmark:before{content:"\f66f"}.fa-language:before{content:"\f1ab"}.fa-laptop:before{content:"\f109"}.fa-laptop-code:before{content:"\f5fc"}.fa-laptop-house:before{content:"\e066"}.fa-laptop-medical:before{content:"\f812"}.fa-laravel:before{content:"\f3bd"}.fa-lastfm:before{content:"\f202"}.fa-lastfm-square:before{content:"\f203"}.fa-laugh:before{content:"\f599"}.fa-laugh-beam:before{content:"\f59a"}.fa-laugh-squint:before{content:"\f59b"}.fa-laugh-wink:before{content:"\f59c"}.fa-layer-group:before{content:"\f5fd"}.fa-leaf:before{content:"\f06c"}.fa-leanpub:before{content:"\f212"}.fa-lemon:before{content:"\f094"}.fa-less:before{content:"\f41d"}.fa-less-than:before{content:"\f536"}.fa-less-than-equal:before{content:"\f537"}.fa-level-down-alt:before{content:"\f3be"}.fa-level-up-alt:before{content:"\f3bf"}.fa-life-ring:before{content:"\f1cd"}.fa-lightbulb:before{content:"\f0eb"}.fa-line:before{content:"\f3c0"}.fa-link:before{content:"\f0c1"}.fa-linkedin:before{content:"\f08c"}.fa-linkedin-in:before{content:"\f0e1"}.fa-linode:before{content:"\f2b8"}.fa-linux:before{content:"\f17c"}.fa-lira-sign:before{content:"\f195"}.fa-list:before{content:"\f03a"}.fa-list-alt:before{content:"\f022"}.fa-list-ol:before{content:"\f0cb"}.fa-list-ul:before{content:"\f0ca"}.fa-location-arrow:before{content:"\f124"}.fa-lock:before{content:"\f023"}.fa-lock-open:before{content:"\f3c1"}.fa-long-arrow-alt-down:before{content:"\f309"}.fa-long-arrow-alt-left:before{content:"\f30a"}.fa-long-arrow-alt-right:before{content:"\f30b"}.fa-long-arrow-alt-up:before{content:"\f30c"}.fa-low-vision:before{content:"\f2a8"}.fa-luggage-cart:before{content:"\f59d"}.fa-lungs:before{content:"\f604"}.fa-lungs-virus:before{content:"\e067"}.fa-lyft:before{content:"\f3c3"}.fa-magento:before{content:"\f3c4"}.fa-magic:before{content:"\f0d0"}.fa-magnet:before{content:"\f076"}.fa-mail-bulk:before{content:"\f674"}.fa-mailchimp:before{content:"\f59e"}.fa-male:before{content:"\f183"}.fa-mandalorian:before{content:"\f50f"}.fa-map:before{content:"\f279"}.fa-map-marked:before{content:"\f59f"}.fa-map-marked-alt:before{content:"\f5a0"}.fa-map-marker:before{content:"\f041"}.fa-map-marker-alt:before{content:"\f3c5"}.fa-map-pin:before{content:"\f276"}.fa-map-signs:before{content:"\f277"}.fa-markdown:before{content:"\f60f"}.fa-marker:before{content:"\f5a1"}.fa-mars:before{content:"\f222"}.fa-mars-double:before{content:"\f227"}.fa-mars-stroke:before{content:"\f229"}.fa-mars-stroke-h:before{content:"\f22b"}.fa-mars-stroke-v:before{content:"\f22a"}.fa-mask:before{content:"\f6fa"}.fa-mastodon:before{content:"\f4f6"}.fa-maxcdn:before{content:"\f136"}.fa-mdb:before{content:"\f8ca"}.fa-medal:before{content:"\f5a2"}.fa-medapps:before{content:"\f3c6"}.fa-medium:before{content:"\f23a"}.fa-medium-m:before{content:"\f3c7"}.fa-medkit:before{content:"\f0fa"}.fa-medrt:before{content:"\f3c8"}.fa-meetup:before{content:"\f2e0"}.fa-megaport:before{content:"\f5a3"}.fa-meh:before{content:"\f11a"}.fa-meh-blank:before{content:"\f5a4"}.fa-meh-rolling-eyes:before{content:"\f5a5"}.fa-memory:before{content:"\f538"}.fa-mendeley:before{content:"\f7b3"}.fa-menorah:before{content:"\f676"}.fa-mercury:before{content:"\f223"}.fa-meteor:before{content:"\f753"}.fa-microblog:before{content:"\e01a"}.fa-microchip:before{content:"\f2db"}.fa-microphone:before{content:"\f130"}.fa-microphone-alt:before{content:"\f3c9"}.fa-microphone-alt-slash:before{content:"\f539"}.fa-microphone-slash:before{content:"\f131"}.fa-microscope:before{content:"\f610"}.fa-microsoft:before{content:"\f3ca"}.fa-minus:before{content:"\f068"}.fa-minus-circle:before{content:"\f056"}.fa-minus-square:before{content:"\f146"}.fa-mitten:before{content:"\f7b5"}.fa-mix:before{content:"\f3cb"}.fa-mixcloud:before{content:"\f289"}.fa-mixer:before{content:"\e056"}.fa-mizuni:before{content:"\f3cc"}.fa-mobile:before{content:"\f10b"}.fa-mobile-alt:before{content:"\f3cd"}.fa-modx:before{content:"\f285"}.fa-monero:before{content:"\f3d0"}.fa-money-bill:before{content:"\f0d6"}.fa-money-bill-alt:before{content:"\f3d1"}.fa-money-bill-wave:before{content:"\f53a"}.fa-money-bill-wave-alt:before{content:"\f53b"}.fa-money-check:before{content:"\f53c"}.fa-money-check-alt:before{content:"\f53d"}.fa-monument:before{content:"\f5a6"}.fa-moon:before{content:"\f186"}.fa-mortar-pestle:before{content:"\f5a7"}.fa-mosque:before{content:"\f678"}.fa-motorcycle:before{content:"\f21c"}.fa-mountain:before{content:"\f6fc"}.fa-mouse:before{content:"\f8cc"}.fa-mouse-pointer:before{content:"\f245"}.fa-mug-hot:before{content:"\f7b6"}.fa-music:before{content:"\f001"}.fa-napster:before{content:"\f3d2"}.fa-neos:before{content:"\f612"}.fa-network-wired:before{content:"\f6ff"}.fa-neuter:before{content:"\f22c"}.fa-newspaper:before{content:"\f1ea"}.fa-nimblr:before{content:"\f5a8"}.fa-node:before{content:"\f419"}.fa-node-js:before{content:"\f3d3"}.fa-not-equal:before{content:"\f53e"}.fa-notes-medical:before{content:"\f481"}.fa-npm:before{content:"\f3d4"}.fa-ns8:before{content:"\f3d5"}.fa-nutritionix:before{content:"\f3d6"}.fa-object-group:before{content:"\f247"}.fa-object-ungroup:before{content:"\f248"}.fa-octopus-deploy:before{content:"\e082"}.fa-odnoklassniki:before{content:"\f263"}.fa-odnoklassniki-square:before{content:"\f264"}.fa-oil-can:before{content:"\f613"}.fa-old-republic:before{content:"\f510"}.fa-om:before{content:"\f679"}.fa-opencart:before{content:"\f23d"}.fa-openid:before{content:"\f19b"}.fa-opera:before{content:"\f26a"}.fa-optin-monster:before{content:"\f23c"}.fa-orcid:before{content:"\f8d2"}.fa-osi:before{content:"\f41a"}.fa-otter:before{content:"\f700"}.fa-outdent:before{content:"\f03b"}.fa-page4:before{content:"\f3d7"}.fa-pagelines:before{content:"\f18c"}.fa-pager:before{content:"\f815"}.fa-paint-brush:before{content:"\f1fc"}.fa-paint-roller:before{content:"\f5aa"}.fa-palette:before{content:"\f53f"}.fa-palfed:before{content:"\f3d8"}.fa-pallet:before{content:"\f482"}.fa-paper-plane:before{content:"\f1d8"}.fa-paperclip:before{content:"\f0c6"}.fa-parachute-box:before{content:"\f4cd"}.fa-paragraph:before{content:"\f1dd"}.fa-parking:before{content:"\f540"}.fa-passport:before{content:"\f5ab"}.fa-pastafarianism:before{content:"\f67b"}.fa-paste:before{content:"\f0ea"}.fa-patreon:before{content:"\f3d9"}.fa-pause:before{content:"\f04c"}.fa-pause-circle:before{content:"\f28b"}.fa-paw:before{content:"\f1b0"}.fa-paypal:before{content:"\f1ed"}.fa-peace:before{content:"\f67c"}.fa-pen:before{content:"\f304"}.fa-pen-alt:before{content:"\f305"}.fa-pen-fancy:before{content:"\f5ac"}.fa-pen-nib:before{content:"\f5ad"}.fa-pen-square:before{content:"\f14b"}.fa-pencil-alt:before{content:"\f303"}.fa-pencil-ruler:before{content:"\f5ae"}.fa-penny-arcade:before{content:"\f704"}.fa-people-arrows:before{content:"\e068"}.fa-people-carry:before{content:"\f4ce"}.fa-pepper-hot:before{content:"\f816"}.fa-perbyte:before{content:"\e083"}.fa-percent:before{content:"\f295"}.fa-percentage:before{content:"\f541"}.fa-periscope:before{content:"\f3da"}.fa-person-booth:before{content:"\f756"}.fa-phabricator:before{content:"\f3db"}.fa-phoenix-framework:before{content:"\f3dc"}.fa-phoenix-squadron:before{content:"\f511"}.fa-phone:before{content:"\f095"}.fa-phone-alt:before{content:"\f879"}.fa-phone-slash:before{content:"\f3dd"}.fa-phone-square:before{content:"\f098"}.fa-phone-square-alt:before{content:"\f87b"}.fa-phone-volume:before{content:"\f2a0"}.fa-photo-video:before{content:"\f87c"}.fa-php:before{content:"\f457"}.fa-pied-piper:before{content:"\f2ae"}.fa-pied-piper-alt:before{content:"\f1a8"}.fa-pied-piper-hat:before{content:"\f4e5"}.fa-pied-piper-pp:before{content:"\f1a7"}.fa-pied-piper-square:before{content:"\e01e"}.fa-piggy-bank:before{content:"\f4d3"}.fa-pills:before{content:"\f484"}.fa-pinterest:before{content:"\f0d2"}.fa-pinterest-p:before{content:"\f231"}.fa-pinterest-square:before{content:"\f0d3"}.fa-pizza-slice:before{content:"\f818"}.fa-place-of-worship:before{content:"\f67f"}.fa-plane:before{content:"\f072"}.fa-plane-arrival:before{content:"\f5af"}.fa-plane-departure:before{content:"\f5b0"}.fa-plane-slash:before{content:"\e069"}.fa-play:before{content:"\f04b"}.fa-play-circle:before{content:"\f144"}.fa-playstation:before{content:"\f3df"}.fa-plug:before{content:"\f1e6"}.fa-plus:before{content:"\f067"}.fa-plus-circle:before{content:"\f055"}.fa-plus-square:before{content:"\f0fe"}.fa-podcast:before{content:"\f2ce"}.fa-poll:before{content:"\f681"}.fa-poll-h:before{content:"\f682"}.fa-poo:before{content:"\f2fe"}.fa-poo-storm:before{content:"\f75a"}.fa-poop:before{content:"\f619"}.fa-portrait:before{content:"\f3e0"}.fa-pound-sign:before{content:"\f154"}.fa-power-off:before{content:"\f011"}.fa-pray:before{content:"\f683"}.fa-praying-hands:before{content:"\f684"}.fa-prescription:before{content:"\f5b1"}.fa-prescription-bottle:before{content:"\f485"}.fa-prescription-bottle-alt:before{content:"\f486"}.fa-print:before{content:"\f02f"}.fa-procedures:before{content:"\f487"}.fa-product-hunt:before{content:"\f288"}.fa-project-diagram:before{content:"\f542"}.fa-pump-medical:before{content:"\e06a"}.fa-pump-soap:before{content:"\e06b"}.fa-pushed:before{content:"\f3e1"}.fa-puzzle-piece:before{content:"\f12e"}.fa-python:before{content:"\f3e2"}.fa-qq:before{content:"\f1d6"}.fa-qrcode:before{content:"\f029"}.fa-question:before{content:"\f128"}.fa-question-circle:before{content:"\f059"}.fa-quidditch:before{content:"\f458"}.fa-quinscape:before{content:"\f459"}.fa-quora:before{content:"\f2c4"}.fa-quote-left:before{content:"\f10d"}.fa-quote-right:before{content:"\f10e"}.fa-quran:before{content:"\f687"}.fa-r-project:before{content:"\f4f7"}.fa-radiation:before{content:"\f7b9"}.fa-radiation-alt:before{content:"\f7ba"}.fa-rainbow:before{content:"\f75b"}.fa-random:before{content:"\f074"}.fa-raspberry-pi:before{content:"\f7bb"}.fa-ravelry:before{content:"\f2d9"}.fa-react:before{content:"\f41b"}.fa-reacteurope:before{content:"\f75d"}.fa-readme:before{content:"\f4d5"}.fa-rebel:before{content:"\f1d0"}.fa-receipt:before{content:"\f543"}.fa-record-vinyl:before{content:"\f8d9"}.fa-recycle:before{content:"\f1b8"}.fa-red-river:before{content:"\f3e3"}.fa-reddit:before{content:"\f1a1"}.fa-reddit-alien:before{content:"\f281"}.fa-reddit-square:before{content:"\f1a2"}.fa-redhat:before{content:"\f7bc"}.fa-redo:before{content:"\f01e"}.fa-redo-alt:before{content:"\f2f9"}.fa-registered:before{content:"\f25d"}.fa-remove-format:before{content:"\f87d"}.fa-renren:before{content:"\f18b"}.fa-reply:before{content:"\f3e5"}.fa-reply-all:before{content:"\f122"}.fa-replyd:before{content:"\f3e6"}.fa-republican:before{content:"\f75e"}.fa-researchgate:before{content:"\f4f8"}.fa-resolving:before{content:"\f3e7"}.fa-restroom:before{content:"\f7bd"}.fa-retweet:before{content:"\f079"}.fa-rev:before{content:"\f5b2"}.fa-ribbon:before{content:"\f4d6"}.fa-ring:before{content:"\f70b"}.fa-road:before{content:"\f018"}.fa-robot:before{content:"\f544"}.fa-rocket:before{content:"\f135"}.fa-rocketchat:before{content:"\f3e8"}.fa-rockrms:before{content:"\f3e9"}.fa-route:before{content:"\f4d7"}.fa-rss:before{content:"\f09e"}.fa-rss-square:before{content:"\f143"}.fa-ruble-sign:before{content:"\f158"}.fa-ruler:before{content:"\f545"}.fa-ruler-combined:before{content:"\f546"}.fa-ruler-horizontal:before{content:"\f547"}.fa-ruler-vertical:before{content:"\f548"}.fa-running:before{content:"\f70c"}.fa-rupee-sign:before{content:"\f156"}.fa-rust:before{content:"\e07a"}.fa-sad-cry:before{content:"\f5b3"}.fa-sad-tear:before{content:"\f5b4"}.fa-safari:before{content:"\f267"}.fa-salesforce:before{content:"\f83b"}.fa-sass:before{content:"\f41e"}.fa-satellite:before{content:"\f7bf"}.fa-satellite-dish:before{content:"\f7c0"}.fa-save:before{content:"\f0c7"}.fa-schlix:before{content:"\f3ea"}.fa-school:before{content:"\f549"}.fa-screwdriver:before{content:"\f54a"}.fa-scribd:before{content:"\f28a"}.fa-scroll:before{content:"\f70e"}.fa-sd-card:before{content:"\f7c2"}.fa-search:before{content:"\f002"}.fa-search-dollar:before{content:"\f688"}.fa-search-location:before{content:"\f689"}.fa-search-minus:before{content:"\f010"}.fa-search-plus:before{content:"\f00e"}.fa-searchengin:before{content:"\f3eb"}.fa-seedling:before{content:"\f4d8"}.fa-sellcast:before{content:"\f2da"}.fa-sellsy:before{content:"\f213"}.fa-server:before{content:"\f233"}.fa-servicestack:before{content:"\f3ec"}.fa-shapes:before{content:"\f61f"}.fa-share:before{content:"\f064"}.fa-share-alt:before{content:"\f1e0"}.fa-share-alt-square:before{content:"\f1e1"}.fa-share-square:before{content:"\f14d"}.fa-shekel-sign:before{content:"\f20b"}.fa-shield-alt:before{content:"\f3ed"}.fa-shield-virus:before{content:"\e06c"}.fa-ship:before{content:"\f21a"}.fa-shipping-fast:before{content:"\f48b"}.fa-shirtsinbulk:before{content:"\f214"}.fa-shoe-prints:before{content:"\f54b"}.fa-shopify:before{content:"\e057"}.fa-shopping-bag:before{content:"\f290"}.fa-shopping-basket:before{content:"\f291"}.fa-shopping-cart:before{content:"\f07a"}.fa-shopware:before{content:"\f5b5"}.fa-shower:before{content:"\f2cc"}.fa-shuttle-van:before{content:"\f5b6"}.fa-sign:before{content:"\f4d9"}.fa-sign-in-alt:before{content:"\f2f6"}.fa-sign-language:before{content:"\f2a7"}.fa-sign-out-alt:before{content:"\f2f5"}.fa-signal:before{content:"\f012"}.fa-signature:before{content:"\f5b7"}.fa-sim-card:before{content:"\f7c4"}.fa-simplybuilt:before{content:"\f215"}.fa-sink:before{content:"\e06d"}.fa-sistrix:before{content:"\f3ee"}.fa-sitemap:before{content:"\f0e8"}.fa-sith:before{content:"\f512"}.fa-skating:before{content:"\f7c5"}.fa-sketch:before{content:"\f7c6"}.fa-skiing:before{content:"\f7c9"}.fa-skiing-nordic:before{content:"\f7ca"}.fa-skull:before{content:"\f54c"}.fa-skull-crossbones:before{content:"\f714"}.fa-skyatlas:before{content:"\f216"}.fa-skype:before{content:"\f17e"}.fa-slack:before{content:"\f198"}.fa-slack-hash:before{content:"\f3ef"}.fa-slash:before{content:"\f715"}.fa-sleigh:before{content:"\f7cc"}.fa-sliders-h:before{content:"\f1de"}.fa-slideshare:before{content:"\f1e7"}.fa-smile:before{content:"\f118"}.fa-smile-beam:before{content:"\f5b8"}.fa-smile-wink:before{content:"\f4da"}.fa-smog:before{content:"\f75f"}.fa-smoking:before{content:"\f48d"}.fa-smoking-ban:before{content:"\f54d"}.fa-sms:before{content:"\f7cd"}.fa-snapchat:before{content:"\f2ab"}.fa-snapchat-ghost:before{content:"\f2ac"}.fa-snapchat-square:before{content:"\f2ad"}.fa-snowboarding:before{content:"\f7ce"}.fa-snowflake:before{content:"\f2dc"}.fa-snowman:before{content:"\f7d0"}.fa-snowplow:before{content:"\f7d2"}.fa-soap:before{content:"\e06e"}.fa-socks:before{content:"\f696"}.fa-solar-panel:before{content:"\f5ba"}.fa-sort:before{content:"\f0dc"}.fa-sort-alpha-down:before{content:"\f15d"}.fa-sort-alpha-down-alt:before{content:"\f881"}.fa-sort-alpha-up:before{content:"\f15e"}.fa-sort-alpha-up-alt:before{content:"\f882"}.fa-sort-amount-down:before{content:"\f160"}.fa-sort-amount-down-alt:before{content:"\f884"}.fa-sort-amount-up:before{content:"\f161"}.fa-sort-amount-up-alt:before{content:"\f885"}.fa-sort-down:before{content:"\f0dd"}.fa-sort-numeric-down:before{content:"\f162"}.fa-sort-numeric-down-alt:before{content:"\f886"}.fa-sort-numeric-up:before{content:"\f163"}.fa-sort-numeric-up-alt:before{content:"\f887"}.fa-sort-up:before{content:"\f0de"}.fa-soundcloud:before{content:"\f1be"}.fa-sourcetree:before{content:"\f7d3"}.fa-spa:before{content:"\f5bb"}.fa-space-shuttle:before{content:"\f197"}.fa-speakap:before{content:"\f3f3"}.fa-speaker-deck:before{content:"\f83c"}.fa-spell-check:before{content:"\f891"}.fa-spider:before{content:"\f717"}.fa-spinner:before{content:"\f110"}.fa-splotch:before{content:"\f5bc"}.fa-spotify:before{content:"\f1bc"}.fa-spray-can:before{content:"\f5bd"}.fa-square:before{content:"\f0c8"}.fa-square-full:before{content:"\f45c"}.fa-square-root-alt:before{content:"\f698"}.fa-squarespace:before{content:"\f5be"}.fa-stack-exchange:before{content:"\f18d"}.fa-stack-overflow:before{content:"\f16c"}.fa-stackpath:before{content:"\f842"}.fa-stamp:before{content:"\f5bf"}.fa-star:before{content:"\f005"}.fa-star-and-crescent:before{content:"\f699"}.fa-star-half:before{content:"\f089"}.fa-star-half-alt:before{content:"\f5c0"}.fa-star-of-david:before{content:"\f69a"}.fa-star-of-life:before{content:"\f621"}.fa-staylinked:before{content:"\f3f5"}.fa-steam:before{content:"\f1b6"}.fa-steam-square:before{content:"\f1b7"}.fa-steam-symbol:before{content:"\f3f6"}.fa-step-backward:before{content:"\f048"}.fa-step-forward:before{content:"\f051"}.fa-stethoscope:before{content:"\f0f1"}.fa-sticker-mule:before{content:"\f3f7"}.fa-sticky-note:before{content:"\f249"}.fa-stop:before{content:"\f04d"}.fa-stop-circle:before{content:"\f28d"}.fa-stopwatch:before{content:"\f2f2"}.fa-stopwatch-20:before{content:"\e06f"}.fa-store:before{content:"\f54e"}.fa-store-alt:before{content:"\f54f"}.fa-store-alt-slash:before{content:"\e070"}.fa-store-slash:before{content:"\e071"}.fa-strava:before{content:"\f428"}.fa-stream:before{content:"\f550"}.fa-street-view:before{content:"\f21d"}.fa-strikethrough:before{content:"\f0cc"}.fa-stripe:before{content:"\f429"}.fa-stripe-s:before{content:"\f42a"}.fa-stroopwafel:before{content:"\f551"}.fa-studiovinari:before{content:"\f3f8"}.fa-stumbleupon:before{content:"\f1a4"}.fa-stumbleupon-circle:before{content:"\f1a3"}.fa-subscript:before{content:"\f12c"}.fa-subway:before{content:"\f239"}.fa-suitcase:before{content:"\f0f2"}.fa-suitcase-rolling:before{content:"\f5c1"}.fa-sun:before{content:"\f185"}.fa-superpowers:before{content:"\f2dd"}.fa-superscript:before{content:"\f12b"}.fa-supple:before{content:"\f3f9"}.fa-surprise:before{content:"\f5c2"}.fa-suse:before{content:"\f7d6"}.fa-swatchbook:before{content:"\f5c3"}.fa-swift:before{content:"\f8e1"}.fa-swimmer:before{content:"\f5c4"}.fa-swimming-pool:before{content:"\f5c5"}.fa-symfony:before{content:"\f83d"}.fa-synagogue:before{content:"\f69b"}.fa-sync:before{content:"\f021"}.fa-sync-alt:before{content:"\f2f1"}.fa-syringe:before{content:"\f48e"}.fa-table:before{content:"\f0ce"}.fa-table-tennis:before{content:"\f45d"}.fa-tablet:before{content:"\f10a"}.fa-tablet-alt:before{content:"\f3fa"}.fa-tablets:before{content:"\f490"}.fa-tachometer-alt:before{content:"\f3fd"}.fa-tag:before{content:"\f02b"}.fa-tags:before{content:"\f02c"}.fa-tape:before{content:"\f4db"}.fa-tasks:before{content:"\f0ae"}.fa-taxi:before{content:"\f1ba"}.fa-teamspeak:before{content:"\f4f9"}.fa-teeth:before{content:"\f62e"}.fa-teeth-open:before{content:"\f62f"}.fa-telegram:before{content:"\f2c6"}.fa-telegram-plane:before{content:"\f3fe"}.fa-temperature-high:before{content:"\f769"}.fa-temperature-low:before{content:"\f76b"}.fa-tencent-weibo:before{content:"\f1d5"}.fa-tenge:before{content:"\f7d7"}.fa-terminal:before{content:"\f120"}.fa-text-height:before{content:"\f034"}.fa-text-width:before{content:"\f035"}.fa-th:before{content:"\f00a"}.fa-th-large:before{content:"\f009"}.fa-th-list:before{content:"\f00b"}.fa-the-red-yeti:before{content:"\f69d"}.fa-theater-masks:before{content:"\f630"}.fa-themeco:before{content:"\f5c6"}.fa-themeisle:before{content:"\f2b2"}.fa-thermometer:before{content:"\f491"}.fa-thermometer-empty:before{content:"\f2cb"}.fa-thermometer-full:before{content:"\f2c7"}.fa-thermometer-half:before{content:"\f2c9"}.fa-thermometer-quarter:before{content:"\f2ca"}.fa-thermometer-three-quarters:before{content:"\f2c8"}.fa-think-peaks:before{content:"\f731"}.fa-thumbs-down:before{content:"\f165"}.fa-thumbs-up:before{content:"\f164"}.fa-thumbtack:before{content:"\f08d"}.fa-ticket-alt:before{content:"\f3ff"}.fa-tiktok:before{content:"\e07b"}.fa-times:before{content:"\f00d"}.fa-times-circle:before{content:"\f057"}.fa-tint:before{content:"\f043"}.fa-tint-slash:before{content:"\f5c7"}.fa-tired:before{content:"\f5c8"}.fa-toggle-off:before{content:"\f204"}.fa-toggle-on:before{content:"\f205"}.fa-toilet:before{content:"\f7d8"}.fa-toilet-paper:before{content:"\f71e"}.fa-toilet-paper-slash:before{content:"\e072"}.fa-toolbox:before{content:"\f552"}.fa-tools:before{content:"\f7d9"}.fa-tooth:before{content:"\f5c9"}.fa-torah:before{content:"\f6a0"}.fa-torii-gate:before{content:"\f6a1"}.fa-tractor:before{content:"\f722"}.fa-trade-federation:before{content:"\f513"}.fa-trademark:before{content:"\f25c"}.fa-traffic-light:before{content:"\f637"}.fa-trailer:before{content:"\e041"}.fa-train:before{content:"\f238"}.fa-tram:before{content:"\f7da"}.fa-transgender:before{content:"\f224"}.fa-transgender-alt:before{content:"\f225"}.fa-trash:before{content:"\f1f8"}.fa-trash-alt:before{content:"\f2ed"}.fa-trash-restore:before{content:"\f829"}.fa-trash-restore-alt:before{content:"\f82a"}.fa-tree:before{content:"\f1bb"}.fa-trello:before{content:"\f181"}.fa-trophy:before{content:"\f091"}.fa-truck:before{content:"\f0d1"}.fa-truck-loading:before{content:"\f4de"}.fa-truck-monster:before{content:"\f63b"}.fa-truck-moving:before{content:"\f4df"}.fa-truck-pickup:before{content:"\f63c"}.fa-tshirt:before{content:"\f553"}.fa-tty:before{content:"\f1e4"}.fa-tumblr:before{content:"\f173"}.fa-tumblr-square:before{content:"\f174"}.fa-tv:before{content:"\f26c"}.fa-twitch:before{content:"\f1e8"}.fa-twitter:before{content:"\f099"}.fa-twitter-square:before{content:"\f081"}.fa-typo3:before{content:"\f42b"}.fa-uber:before{content:"\f402"}.fa-ubuntu:before{content:"\f7df"}.fa-uikit:before{content:"\f403"}.fa-umbraco:before{content:"\f8e8"}.fa-umbrella:before{content:"\f0e9"}.fa-umbrella-beach:before{content:"\f5ca"}.fa-uncharted:before{content:"\e084"}.fa-underline:before{content:"\f0cd"}.fa-undo:before{content:"\f0e2"}.fa-undo-alt:before{content:"\f2ea"}.fa-uniregistry:before{content:"\f404"}.fa-unity:before{content:"\e049"}.fa-universal-access:before{content:"\f29a"}.fa-university:before{content:"\f19c"}.fa-unlink:before{content:"\f127"}.fa-unlock:before{content:"\f09c"}.fa-unlock-alt:before{content:"\f13e"}.fa-unsplash:before{content:"\e07c"}.fa-untappd:before{content:"\f405"}.fa-upload:before{content:"\f093"}.fa-ups:before{content:"\f7e0"}.fa-usb:before{content:"\f287"}.fa-user:before{content:"\f007"}.fa-user-alt:before{content:"\f406"}.fa-user-alt-slash:before{content:"\f4fa"}.fa-user-astronaut:before{content:"\f4fb"}.fa-user-check:before{content:"\f4fc"}.fa-user-circle:before{content:"\f2bd"}.fa-user-clock:before{content:"\f4fd"}.fa-user-cog:before{content:"\f4fe"}.fa-user-edit:before{content:"\f4ff"}.fa-user-friends:before{content:"\f500"}.fa-user-graduate:before{content:"\f501"}.fa-user-injured:before{content:"\f728"}.fa-user-lock:before{content:"\f502"}.fa-user-md:before{content:"\f0f0"}.fa-user-minus:before{content:"\f503"}.fa-user-ninja:before{content:"\f504"}.fa-user-nurse:before{content:"\f82f"}.fa-user-plus:before{content:"\f234"}.fa-user-secret:before{content:"\f21b"}.fa-user-shield:before{content:"\f505"}.fa-user-slash:before{content:"\f506"}.fa-user-tag:before{content:"\f507"}.fa-user-tie:before{content:"\f508"}.fa-user-times:before{content:"\f235"}.fa-users:before{content:"\f0c0"}.fa-users-cog:before{content:"\f509"}.fa-users-slash:before{content:"\e073"}.fa-usps:before{content:"\f7e1"}.fa-ussunnah:before{content:"\f407"}.fa-utensil-spoon:before{content:"\f2e5"}.fa-utensils:before{content:"\f2e7"}.fa-vaadin:before{content:"\f408"}.fa-vector-square:before{content:"\f5cb"}.fa-venus:before{content:"\f221"}.fa-venus-double:before{content:"\f226"}.fa-venus-mars:before{content:"\f228"}.fa-vest:before{content:"\e085"}.fa-vest-patches:before{content:"\e086"}.fa-viacoin:before{content:"\f237"}.fa-viadeo:before{content:"\f2a9"}.fa-viadeo-square:before{content:"\f2aa"}.fa-vial:before{content:"\f492"}.fa-vials:before{content:"\f493"}.fa-viber:before{content:"\f409"}.fa-video:before{content:"\f03d"}.fa-video-slash:before{content:"\f4e2"}.fa-vihara:before{content:"\f6a7"}.fa-vimeo:before{content:"\f40a"}.fa-vimeo-square:before{content:"\f194"}.fa-vimeo-v:before{content:"\f27d"}.fa-vine:before{content:"\f1ca"}.fa-virus:before{content:"\e074"}.fa-virus-slash:before{content:"\e075"}.fa-viruses:before{content:"\e076"}.fa-vk:before{content:"\f189"}.fa-vnv:before{content:"\f40b"}.fa-voicemail:before{content:"\f897"}.fa-volleyball-ball:before{content:"\f45f"}.fa-volume-down:before{content:"\f027"}.fa-volume-mute:before{content:"\f6a9"}.fa-volume-off:before{content:"\f026"}.fa-volume-up:before{content:"\f028"}.fa-vote-yea:before{content:"\f772"}.fa-vr-cardboard:before{content:"\f729"}.fa-vuejs:before{content:"\f41f"}.fa-walking:before{content:"\f554"}.fa-wallet:before{content:"\f555"}.fa-warehouse:before{content:"\f494"}.fa-watchman-monitoring:before{content:"\e087"}.fa-water:before{content:"\f773"}.fa-wave-square:before{content:"\f83e"}.fa-waze:before{content:"\f83f"}.fa-weebly:before{content:"\f5cc"}.fa-weibo:before{content:"\f18a"}.fa-weight:before{content:"\f496"}.fa-weight-hanging:before{content:"\f5cd"}.fa-weixin:before{content:"\f1d7"}.fa-whatsapp:before{content:"\f232"}.fa-whatsapp-square:before{content:"\f40c"}.fa-wheelchair:before{content:"\f193"}.fa-whmcs:before{content:"\f40d"}.fa-wifi:before{content:"\f1eb"}.fa-wikipedia-w:before{content:"\f266"}.fa-wind:before{content:"\f72e"}.fa-window-close:before{content:"\f410"}.fa-window-maximize:before{content:"\f2d0"}.fa-window-minimize:before{content:"\f2d1"}.fa-window-restore:before{content:"\f2d2"}.fa-windows:before{content:"\f17a"}.fa-wine-bottle:before{content:"\f72f"}.fa-wine-glass:before{content:"\f4e3"}.fa-wine-glass-alt:before{content:"\f5ce"}.fa-wix:before{content:"\f5cf"}.fa-wizards-of-the-coast:before{content:"\f730"}.fa-wodu:before{content:"\e088"}.fa-wolf-pack-battalion:before{content:"\f514"}.fa-won-sign:before{content:"\f159"}.fa-wordpress:before{content:"\f19a"}.fa-wordpress-simple:before{content:"\f411"}.fa-wpbeginner:before{content:"\f297"}.fa-wpexplorer:before{content:"\f2de"}.fa-wpforms:before{content:"\f298"}.fa-wpressr:before{content:"\f3e4"}.fa-wrench:before{content:"\f0ad"}.fa-x-ray:before{content:"\f497"}.fa-xbox:before{content:"\f412"}.fa-xing:before{content:"\f168"}.fa-xing-square:before{content:"\f169"}.fa-y-combinator:before{content:"\f23b"}.fa-yahoo:before{content:"\f19e"}.fa-yammer:before{content:"\f840"}.fa-yandex:before{content:"\f413"}.fa-yandex-international:before{content:"\f414"}.fa-yarn:before{content:"\f7e3"}.fa-yelp:before{content:"\f1e9"}.fa-yen-sign:before{content:"\f157"}.fa-yin-yang:before{content:"\f6ad"}.fa-yoast:before{content:"\f2b1"}.fa-youtube:before{content:"\f167"}.fa-youtube-square:before{content:"\f431"}.fa-zhihu:before{content:"\f63f"}.sr-only{border:0;clip:rect(0,0,0,0);height:1px;margin:-1px;overflow:hidden;padding:0;position:absolute;width:1px}.sr-only-focusable:active,.sr-only-focusable:focus{clip:auto;height:auto;margin:0;overflow:visible;position:static;width:auto}@font-face{font-family:"Font Awesome 5 Brands";font-style:normal;font-weight:400;font-display:block;src:url(http://localhost:8888/static/notebook/e4299464e7b012968eed.eot);src:url(http://localhost:8888/static/notebook/e4299464e7b012968eed.eot?#iefix) format("embedded-opentype"),url(http://localhost:8888/static/notebook/8ea8791754915a898a31.woff2) format("woff2"),url(http://localhost:8888/static/notebook/f9217f66874b0c01cd8c.woff) format("woff"),url(http://localhost:8888/static/notebook/cda59d6efffa685830fd.ttf) format("truetype"),url(http://localhost:8888/static/notebook/a3b9817780214caf01e8.svg#fontawesome) format("svg")}.fab{font-family:"Font Awesome 5 Brands"}@font-face{font-family:"Font Awesome 5 Free";font-style:normal;font-weight:400;font-display:block;src:url(http://localhost:8888/static/notebook/79d088064beb3826054f.eot);src:url(http://localhost:8888/static/notebook/79d088064beb3826054f.eot?#iefix) format("embedded-opentype"),url(http://localhost:8888/static/notebook/e42a88444448ac3d6054.woff2) format("woff2"),url(http://localhost:8888/static/notebook/cb9e9e693192413cde2b.woff) format("woff"),url(http://localhost:8888/static/notebook/e8711bbb871afd8e9dea.ttf) format("truetype"),url(http://localhost:8888/static/notebook/be0a084962d8066884f7.svg#fontawesome) format("svg")}.fab,.far{font-weight:400}@font-face{font-family:"Font Awesome 5 Free";font-style:normal;font-weight:900;font-display:block;src:url(http://localhost:8888/static/notebook/373c04fd2418f5c77eea.eot);src:url(http://localhost:8888/static/notebook/373c04fd2418f5c77eea.eot?#iefix) format("embedded-opentype"),url(http://localhost:8888/static/notebook/9834b82ad26e2a37583d.woff2) format("woff2"),url(http://localhost:8888/static/notebook/3f6d3488cf65374f6f67.woff) format("woff"),url(http://localhost:8888/static/notebook/af6397503fcefbd61397.ttf) format("truetype"),url(http://localhost:8888/static/notebook/9674eb1bd55047179038.svg#fontawesome) format("svg")}.fa,.far,.fas{font-family:"Font Awesome 5 Free"}.fa,.fas{font-weight:900}</style><style>/*!
 * Font Awesome Free 5.15.4 by @fontawesome - https://fontawesome.com
 * License - https://fontawesome.com/license/free (Icons: CC BY 4.0, Fonts: SIL OFL 1.1, Code: MIT License)
 */
.fa.fa-glass:before{content:"\f000"}.fa.fa-meetup{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-star-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-star-o:before{content:"\f005"}.fa.fa-close:before,.fa.fa-remove:before{content:"\f00d"}.fa.fa-gear:before{content:"\f013"}.fa.fa-trash-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-trash-o:before{content:"\f2ed"}.fa.fa-file-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-o:before{content:"\f15b"}.fa.fa-clock-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-clock-o:before{content:"\f017"}.fa.fa-arrow-circle-o-down{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-arrow-circle-o-down:before{content:"\f358"}.fa.fa-arrow-circle-o-up{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-arrow-circle-o-up:before{content:"\f35b"}.fa.fa-play-circle-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-play-circle-o:before{content:"\f144"}.fa.fa-repeat:before,.fa.fa-rotate-right:before{content:"\f01e"}.fa.fa-refresh:before{content:"\f021"}.fa.fa-list-alt{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-dedent:before{content:"\f03b"}.fa.fa-video-camera:before{content:"\f03d"}.fa.fa-picture-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-picture-o:before{content:"\f03e"}.fa.fa-photo{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-photo:before{content:"\f03e"}.fa.fa-image{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-image:before{content:"\f03e"}.fa.fa-pencil:before{content:"\f303"}.fa.fa-map-marker:before{content:"\f3c5"}.fa.fa-pencil-square-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-pencil-square-o:before{content:"\f044"}.fa.fa-share-square-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-share-square-o:before{content:"\f14d"}.fa.fa-check-square-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-check-square-o:before{content:"\f14a"}.fa.fa-arrows:before{content:"\f0b2"}.fa.fa-times-circle-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-times-circle-o:before{content:"\f057"}.fa.fa-check-circle-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-check-circle-o:before{content:"\f058"}.fa.fa-mail-forward:before{content:"\f064"}.fa.fa-expand:before{content:"\f424"}.fa.fa-compress:before{content:"\f422"}.fa.fa-eye,.fa.fa-eye-slash{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-warning:before{content:"\f071"}.fa.fa-calendar:before{content:"\f073"}.fa.fa-arrows-v:before{content:"\f338"}.fa.fa-arrows-h:before{content:"\f337"}.fa.fa-bar-chart{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-bar-chart:before{content:"\f080"}.fa.fa-bar-chart-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-bar-chart-o:before{content:"\f080"}.fa.fa-facebook-square,.fa.fa-twitter-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-gears:before{content:"\f085"}.fa.fa-thumbs-o-up{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-thumbs-o-up:before{content:"\f164"}.fa.fa-thumbs-o-down{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-thumbs-o-down:before{content:"\f165"}.fa.fa-heart-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-heart-o:before{content:"\f004"}.fa.fa-sign-out:before{content:"\f2f5"}.fa.fa-linkedin-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-linkedin-square:before{content:"\f08c"}.fa.fa-thumb-tack:before{content:"\f08d"}.fa.fa-external-link:before{content:"\f35d"}.fa.fa-sign-in:before{content:"\f2f6"}.fa.fa-github-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-lemon-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-lemon-o:before{content:"\f094"}.fa.fa-square-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-square-o:before{content:"\f0c8"}.fa.fa-bookmark-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-bookmark-o:before{content:"\f02e"}.fa.fa-facebook,.fa.fa-twitter{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-facebook:before{content:"\f39e"}.fa.fa-facebook-f{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-facebook-f:before{content:"\f39e"}.fa.fa-github{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-credit-card{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-feed:before{content:"\f09e"}.fa.fa-hdd-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hdd-o:before{content:"\f0a0"}.fa.fa-hand-o-right{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-o-right:before{content:"\f0a4"}.fa.fa-hand-o-left{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-o-left:before{content:"\f0a5"}.fa.fa-hand-o-up{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-o-up:before{content:"\f0a6"}.fa.fa-hand-o-down{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-o-down:before{content:"\f0a7"}.fa.fa-arrows-alt:before{content:"\f31e"}.fa.fa-group:before{content:"\f0c0"}.fa.fa-chain:before{content:"\f0c1"}.fa.fa-scissors:before{content:"\f0c4"}.fa.fa-files-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-files-o:before{content:"\f0c5"}.fa.fa-floppy-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-floppy-o:before{content:"\f0c7"}.fa.fa-navicon:before,.fa.fa-reorder:before{content:"\f0c9"}.fa.fa-google-plus,.fa.fa-google-plus-square,.fa.fa-pinterest,.fa.fa-pinterest-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-google-plus:before{content:"\f0d5"}.fa.fa-money{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-money:before{content:"\f3d1"}.fa.fa-unsorted:before{content:"\f0dc"}.fa.fa-sort-desc:before{content:"\f0dd"}.fa.fa-sort-asc:before{content:"\f0de"}.fa.fa-linkedin{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-linkedin:before{content:"\f0e1"}.fa.fa-rotate-left:before{content:"\f0e2"}.fa.fa-legal:before{content:"\f0e3"}.fa.fa-dashboard:before,.fa.fa-tachometer:before{content:"\f3fd"}.fa.fa-comment-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-comment-o:before{content:"\f075"}.fa.fa-comments-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-comments-o:before{content:"\f086"}.fa.fa-flash:before{content:"\f0e7"}.fa.fa-clipboard,.fa.fa-paste{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-paste:before{content:"\f328"}.fa.fa-lightbulb-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-lightbulb-o:before{content:"\f0eb"}.fa.fa-exchange:before{content:"\f362"}.fa.fa-cloud-download:before{content:"\f381"}.fa.fa-cloud-upload:before{content:"\f382"}.fa.fa-bell-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-bell-o:before{content:"\f0f3"}.fa.fa-cutlery:before{content:"\f2e7"}.fa.fa-file-text-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-text-o:before{content:"\f15c"}.fa.fa-building-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-building-o:before{content:"\f1ad"}.fa.fa-hospital-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hospital-o:before{content:"\f0f8"}.fa.fa-tablet:before{content:"\f3fa"}.fa.fa-mobile-phone:before,.fa.fa-mobile:before{content:"\f3cd"}.fa.fa-circle-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-circle-o:before{content:"\f111"}.fa.fa-mail-reply:before{content:"\f3e5"}.fa.fa-github-alt{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-folder-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-folder-o:before{content:"\f07b"}.fa.fa-folder-open-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-folder-open-o:before{content:"\f07c"}.fa.fa-smile-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-smile-o:before{content:"\f118"}.fa.fa-frown-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-frown-o:before{content:"\f119"}.fa.fa-meh-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-meh-o:before{content:"\f11a"}.fa.fa-keyboard-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-keyboard-o:before{content:"\f11c"}.fa.fa-flag-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-flag-o:before{content:"\f024"}.fa.fa-mail-reply-all:before{content:"\f122"}.fa.fa-star-half-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-star-half-o:before{content:"\f089"}.fa.fa-star-half-empty{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-star-half-empty:before{content:"\f089"}.fa.fa-star-half-full{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-star-half-full:before{content:"\f089"}.fa.fa-code-fork:before{content:"\f126"}.fa.fa-chain-broken:before{content:"\f127"}.fa.fa-shield:before{content:"\f3ed"}.fa.fa-calendar-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-calendar-o:before{content:"\f133"}.fa.fa-css3,.fa.fa-html5,.fa.fa-maxcdn{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-ticket:before{content:"\f3ff"}.fa.fa-minus-square-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-minus-square-o:before{content:"\f146"}.fa.fa-level-up:before{content:"\f3bf"}.fa.fa-level-down:before{content:"\f3be"}.fa.fa-pencil-square:before{content:"\f14b"}.fa.fa-external-link-square:before{content:"\f360"}.fa.fa-compass{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-caret-square-o-down{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-caret-square-o-down:before{content:"\f150"}.fa.fa-toggle-down{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-toggle-down:before{content:"\f150"}.fa.fa-caret-square-o-up{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-caret-square-o-up:before{content:"\f151"}.fa.fa-toggle-up{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-toggle-up:before{content:"\f151"}.fa.fa-caret-square-o-right{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-caret-square-o-right:before{content:"\f152"}.fa.fa-toggle-right{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-toggle-right:before{content:"\f152"}.fa.fa-eur:before,.fa.fa-euro:before{content:"\f153"}.fa.fa-gbp:before{content:"\f154"}.fa.fa-dollar:before,.fa.fa-usd:before{content:"\f155"}.fa.fa-inr:before,.fa.fa-rupee:before{content:"\f156"}.fa.fa-cny:before,.fa.fa-jpy:before,.fa.fa-rmb:before,.fa.fa-yen:before{content:"\f157"}.fa.fa-rouble:before,.fa.fa-rub:before,.fa.fa-ruble:before{content:"\f158"}.fa.fa-krw:before,.fa.fa-won:before{content:"\f159"}.fa.fa-bitcoin,.fa.fa-btc{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-bitcoin:before{content:"\f15a"}.fa.fa-file-text:before{content:"\f15c"}.fa.fa-sort-alpha-asc:before{content:"\f15d"}.fa.fa-sort-alpha-desc:before{content:"\f881"}.fa.fa-sort-amount-asc:before{content:"\f160"}.fa.fa-sort-amount-desc:before{content:"\f884"}.fa.fa-sort-numeric-asc:before{content:"\f162"}.fa.fa-sort-numeric-desc:before{content:"\f886"}.fa.fa-xing,.fa.fa-xing-square,.fa.fa-youtube,.fa.fa-youtube-play,.fa.fa-youtube-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-youtube-play:before{content:"\f167"}.fa.fa-adn,.fa.fa-bitbucket,.fa.fa-bitbucket-square,.fa.fa-dropbox,.fa.fa-flickr,.fa.fa-instagram,.fa.fa-stack-overflow{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-bitbucket-square:before{content:"\f171"}.fa.fa-tumblr,.fa.fa-tumblr-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-long-arrow-down:before{content:"\f309"}.fa.fa-long-arrow-up:before{content:"\f30c"}.fa.fa-long-arrow-left:before{content:"\f30a"}.fa.fa-long-arrow-right:before{content:"\f30b"}.fa.fa-android,.fa.fa-apple,.fa.fa-dribbble,.fa.fa-foursquare,.fa.fa-gittip,.fa.fa-gratipay,.fa.fa-linux,.fa.fa-skype,.fa.fa-trello,.fa.fa-windows{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-gittip:before{content:"\f184"}.fa.fa-sun-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-sun-o:before{content:"\f185"}.fa.fa-moon-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-moon-o:before{content:"\f186"}.fa.fa-pagelines,.fa.fa-renren,.fa.fa-stack-exchange,.fa.fa-vk,.fa.fa-weibo{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-arrow-circle-o-right{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-arrow-circle-o-right:before{content:"\f35a"}.fa.fa-arrow-circle-o-left{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-arrow-circle-o-left:before{content:"\f359"}.fa.fa-caret-square-o-left{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-caret-square-o-left:before{content:"\f191"}.fa.fa-toggle-left{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-toggle-left:before{content:"\f191"}.fa.fa-dot-circle-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-dot-circle-o:before{content:"\f192"}.fa.fa-vimeo-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-try:before,.fa.fa-turkish-lira:before{content:"\f195"}.fa.fa-plus-square-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-plus-square-o:before{content:"\f0fe"}.fa.fa-openid,.fa.fa-slack,.fa.fa-wordpress{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-bank:before,.fa.fa-institution:before{content:"\f19c"}.fa.fa-mortar-board:before{content:"\f19d"}.fa.fa-delicious,.fa.fa-digg,.fa.fa-drupal,.fa.fa-google,.fa.fa-joomla,.fa.fa-pied-piper-alt,.fa.fa-pied-piper-pp,.fa.fa-reddit,.fa.fa-reddit-square,.fa.fa-stumbleupon,.fa.fa-stumbleupon-circle,.fa.fa-yahoo{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-spoon:before{content:"\f2e5"}.fa.fa-behance,.fa.fa-behance-square,.fa.fa-steam,.fa.fa-steam-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-automobile:before{content:"\f1b9"}.fa.fa-envelope-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-envelope-o:before{content:"\f0e0"}.fa.fa-deviantart,.fa.fa-soundcloud,.fa.fa-spotify{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-file-pdf-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-pdf-o:before{content:"\f1c1"}.fa.fa-file-word-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-word-o:before{content:"\f1c2"}.fa.fa-file-excel-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-excel-o:before{content:"\f1c3"}.fa.fa-file-powerpoint-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-powerpoint-o:before{content:"\f1c4"}.fa.fa-file-image-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-image-o:before{content:"\f1c5"}.fa.fa-file-photo-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-photo-o:before{content:"\f1c5"}.fa.fa-file-picture-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-picture-o:before{content:"\f1c5"}.fa.fa-file-archive-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-archive-o:before{content:"\f1c6"}.fa.fa-file-zip-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-zip-o:before{content:"\f1c6"}.fa.fa-file-audio-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-audio-o:before{content:"\f1c7"}.fa.fa-file-sound-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-sound-o:before{content:"\f1c7"}.fa.fa-file-video-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-video-o:before{content:"\f1c8"}.fa.fa-file-movie-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-movie-o:before{content:"\f1c8"}.fa.fa-file-code-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-file-code-o:before{content:"\f1c9"}.fa.fa-codepen,.fa.fa-jsfiddle,.fa.fa-vine{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-life-bouy,.fa.fa-life-ring{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-life-bouy:before{content:"\f1cd"}.fa.fa-life-buoy{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-life-buoy:before{content:"\f1cd"}.fa.fa-life-saver{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-life-saver:before{content:"\f1cd"}.fa.fa-support{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-support:before{content:"\f1cd"}.fa.fa-circle-o-notch:before{content:"\f1ce"}.fa.fa-ra,.fa.fa-rebel{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-ra:before{content:"\f1d0"}.fa.fa-resistance{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-resistance:before{content:"\f1d0"}.fa.fa-empire,.fa.fa-ge{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-ge:before{content:"\f1d1"}.fa.fa-git,.fa.fa-git-square,.fa.fa-hacker-news,.fa.fa-y-combinator-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-y-combinator-square:before{content:"\f1d4"}.fa.fa-yc-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-yc-square:before{content:"\f1d4"}.fa.fa-qq,.fa.fa-tencent-weibo,.fa.fa-wechat,.fa.fa-weixin{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-wechat:before{content:"\f1d7"}.fa.fa-send:before{content:"\f1d8"}.fa.fa-paper-plane-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-paper-plane-o:before{content:"\f1d8"}.fa.fa-send-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-send-o:before{content:"\f1d8"}.fa.fa-circle-thin{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-circle-thin:before{content:"\f111"}.fa.fa-header:before{content:"\f1dc"}.fa.fa-sliders:before{content:"\f1de"}.fa.fa-futbol-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-futbol-o:before{content:"\f1e3"}.fa.fa-soccer-ball-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-soccer-ball-o:before{content:"\f1e3"}.fa.fa-slideshare,.fa.fa-twitch,.fa.fa-yelp{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-newspaper-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-newspaper-o:before{content:"\f1ea"}.fa.fa-cc-amex,.fa.fa-cc-discover,.fa.fa-cc-mastercard,.fa.fa-cc-paypal,.fa.fa-cc-stripe,.fa.fa-cc-visa,.fa.fa-google-wallet,.fa.fa-paypal{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-bell-slash-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-bell-slash-o:before{content:"\f1f6"}.fa.fa-trash:before{content:"\f2ed"}.fa.fa-copyright{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-eyedropper:before{content:"\f1fb"}.fa.fa-area-chart:before{content:"\f1fe"}.fa.fa-pie-chart:before{content:"\f200"}.fa.fa-line-chart:before{content:"\f201"}.fa.fa-angellist,.fa.fa-ioxhost,.fa.fa-lastfm,.fa.fa-lastfm-square{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-cc{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-cc:before{content:"\f20a"}.fa.fa-ils:before,.fa.fa-shekel:before,.fa.fa-sheqel:before{content:"\f20b"}.fa.fa-meanpath{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-meanpath:before{content:"\f2b4"}.fa.fa-buysellads,.fa.fa-connectdevelop,.fa.fa-dashcube,.fa.fa-forumbee,.fa.fa-leanpub,.fa.fa-sellsy,.fa.fa-shirtsinbulk,.fa.fa-simplybuilt,.fa.fa-skyatlas{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-diamond{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-diamond:before{content:"\f3a5"}.fa.fa-intersex:before{content:"\f224"}.fa.fa-facebook-official{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-facebook-official:before{content:"\f09a"}.fa.fa-pinterest-p,.fa.fa-whatsapp{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-hotel:before{content:"\f236"}.fa.fa-medium,.fa.fa-viacoin,.fa.fa-y-combinator,.fa.fa-yc{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-yc:before{content:"\f23b"}.fa.fa-expeditedssl,.fa.fa-opencart,.fa.fa-optin-monster{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-battery-4:before,.fa.fa-battery:before{content:"\f240"}.fa.fa-battery-3:before{content:"\f241"}.fa.fa-battery-2:before{content:"\f242"}.fa.fa-battery-1:before{content:"\f243"}.fa.fa-battery-0:before{content:"\f244"}.fa.fa-object-group,.fa.fa-object-ungroup,.fa.fa-sticky-note-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-sticky-note-o:before{content:"\f249"}.fa.fa-cc-diners-club,.fa.fa-cc-jcb{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-clone,.fa.fa-hourglass-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hourglass-o:before{content:"\f254"}.fa.fa-hourglass-1:before{content:"\f251"}.fa.fa-hourglass-2:before{content:"\f252"}.fa.fa-hourglass-3:before{content:"\f253"}.fa.fa-hand-rock-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-rock-o:before{content:"\f255"}.fa.fa-hand-grab-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-grab-o:before{content:"\f255"}.fa.fa-hand-paper-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-paper-o:before{content:"\f256"}.fa.fa-hand-stop-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-stop-o:before{content:"\f256"}.fa.fa-hand-scissors-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-scissors-o:before{content:"\f257"}.fa.fa-hand-lizard-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-lizard-o:before{content:"\f258"}.fa.fa-hand-spock-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-spock-o:before{content:"\f259"}.fa.fa-hand-pointer-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-pointer-o:before{content:"\f25a"}.fa.fa-hand-peace-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-hand-peace-o:before{content:"\f25b"}.fa.fa-registered{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-chrome,.fa.fa-creative-commons,.fa.fa-firefox,.fa.fa-get-pocket,.fa.fa-gg,.fa.fa-gg-circle,.fa.fa-internet-explorer,.fa.fa-odnoklassniki,.fa.fa-odnoklassniki-square,.fa.fa-opera,.fa.fa-safari,.fa.fa-tripadvisor,.fa.fa-wikipedia-w{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-television:before{content:"\f26c"}.fa.fa-500px,.fa.fa-amazon,.fa.fa-contao{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-calendar-plus-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-calendar-plus-o:before{content:"\f271"}.fa.fa-calendar-minus-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-calendar-minus-o:before{content:"\f272"}.fa.fa-calendar-times-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-calendar-times-o:before{content:"\f273"}.fa.fa-calendar-check-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-calendar-check-o:before{content:"\f274"}.fa.fa-map-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-map-o:before{content:"\f279"}.fa.fa-commenting:before{content:"\f4ad"}.fa.fa-commenting-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-commenting-o:before{content:"\f4ad"}.fa.fa-houzz,.fa.fa-vimeo{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-vimeo:before{content:"\f27d"}.fa.fa-black-tie,.fa.fa-edge,.fa.fa-fonticons,.fa.fa-reddit-alien{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-credit-card-alt:before{content:"\f09d"}.fa.fa-codiepie,.fa.fa-fort-awesome,.fa.fa-mixcloud,.fa.fa-modx,.fa.fa-product-hunt,.fa.fa-scribd,.fa.fa-usb{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-pause-circle-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-pause-circle-o:before{content:"\f28b"}.fa.fa-stop-circle-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-stop-circle-o:before{content:"\f28d"}.fa.fa-bluetooth,.fa.fa-bluetooth-b,.fa.fa-envira,.fa.fa-gitlab,.fa.fa-wheelchair-alt,.fa.fa-wpbeginner,.fa.fa-wpforms{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-wheelchair-alt:before{content:"\f368"}.fa.fa-question-circle-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-question-circle-o:before{content:"\f059"}.fa.fa-volume-control-phone:before{content:"\f2a0"}.fa.fa-asl-interpreting:before{content:"\f2a3"}.fa.fa-deafness:before,.fa.fa-hard-of-hearing:before{content:"\f2a4"}.fa.fa-glide,.fa.fa-glide-g{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-signing:before{content:"\f2a7"}.fa.fa-first-order,.fa.fa-google-plus-official,.fa.fa-pied-piper,.fa.fa-snapchat,.fa.fa-snapchat-ghost,.fa.fa-snapchat-square,.fa.fa-themeisle,.fa.fa-viadeo,.fa.fa-viadeo-square,.fa.fa-yoast{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-google-plus-official:before{content:"\f2b3"}.fa.fa-google-plus-circle{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-google-plus-circle:before{content:"\f2b3"}.fa.fa-fa,.fa.fa-font-awesome{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-fa:before{content:"\f2b4"}.fa.fa-handshake-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-handshake-o:before{content:"\f2b5"}.fa.fa-envelope-open-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-envelope-open-o:before{content:"\f2b6"}.fa.fa-linode{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-address-book-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-address-book-o:before{content:"\f2b9"}.fa.fa-vcard:before{content:"\f2bb"}.fa.fa-address-card-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-address-card-o:before{content:"\f2bb"}.fa.fa-vcard-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-vcard-o:before{content:"\f2bb"}.fa.fa-user-circle-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-user-circle-o:before{content:"\f2bd"}.fa.fa-user-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-user-o:before{content:"\f007"}.fa.fa-id-badge{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-drivers-license:before{content:"\f2c2"}.fa.fa-id-card-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-id-card-o:before{content:"\f2c2"}.fa.fa-drivers-license-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-drivers-license-o:before{content:"\f2c2"}.fa.fa-free-code-camp,.fa.fa-quora,.fa.fa-telegram{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-thermometer-4:before,.fa.fa-thermometer:before{content:"\f2c7"}.fa.fa-thermometer-3:before{content:"\f2c8"}.fa.fa-thermometer-2:before{content:"\f2c9"}.fa.fa-thermometer-1:before{content:"\f2ca"}.fa.fa-thermometer-0:before{content:"\f2cb"}.fa.fa-bathtub:before,.fa.fa-s15:before{content:"\f2cd"}.fa.fa-window-maximize,.fa.fa-window-restore{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-times-rectangle:before{content:"\f410"}.fa.fa-window-close-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-window-close-o:before{content:"\f410"}.fa.fa-times-rectangle-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-times-rectangle-o:before{content:"\f410"}.fa.fa-bandcamp,.fa.fa-eercast,.fa.fa-etsy,.fa.fa-grav,.fa.fa-imdb,.fa.fa-ravelry{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-eercast:before{content:"\f2da"}.fa.fa-snowflake-o{font-family:"Font Awesome 5 Free";font-weight:400}.fa.fa-snowflake-o:before{content:"\f2dc"}.fa.fa-superpowers,.fa.fa-wpexplorer{font-family:"Font Awesome 5 Brands";font-weight:400}.fa.fa-cab:before{content:"\f1ba"}</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
}

.lm-Widget.lm-mod-hidden {
  display: none !important;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.lm-AccordionPanel[data-orientation='horizontal'] > .lm-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-CommandPalette-search {
  flex: 0 0 auto;
}

.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}

.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}

.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}

.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
  border: 1px solid transparent;
  background-color: transparent;
  position: absolute;
  z-index: 1;
  right: 3%;
  top: 0;
  bottom: 0;
  margin: auto;
  padding: 7px 0;
  display: none;
  vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
  content: 'X';
  display: block;
  width: 15px;
  height: 15px;
  text-align: center;
  color: #000;
  font-weight: normal;
  font-size: 12px;
  cursor: pointer;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-DockPanel {
  z-index: 0;
}

.lm-DockPanel-widget {
  z-index: 0;
}

.lm-DockPanel-tabBar {
  z-index: 1;
}

.lm-DockPanel-handle {
  z-index: 2;
}

.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}

.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}

.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}

.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}

.lm-Menu-item {
  display: table-row;
}

.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}

.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}

.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}

.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}

.lm-MenuBar-item {
  box-sizing: border-box;
}

.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}

.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}

.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}

.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}

.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-SplitPanel-child {
  z-index: 0;
}

.lm-SplitPanel-handle {
  z-index: 1;
}

.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}

.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}

.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}

.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}

.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}

.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
  touch-action: none; /* Disable native Drag/Drop */
}

.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}

.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}

.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
}

.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}

.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}

.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
  background: inherit;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabPanel-tabBar {
  z-index: 1;
}

.lm-TabPanel-stackedPanel {
  z-index: 0;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.jp-Collapse-header {
  padding: 1px 12px;
  background-color: var(--jp-layout-color1);
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  align-items: center;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  text-transform: uppercase;
  user-select: none;
}

.jp-Collapser-icon {
  height: 16px;
}

.jp-Collapse-header-collapsed .jp-Collapser-icon {
  transform: rotate(-90deg);
  margin: auto 0;
}

.jp-Collapser-title {
  line-height: 25px;
}

.jp-Collapse-contents {
  padding: 0 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add-above: url("data:image/svg+xml,%3csvg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg clip-path='url(%23clip0_137_19492)'%3e %3cpath class='jp-icon3' d='M4.75 4.93066H6.625V6.80566C6.625 7.01191 6.79375 7.18066 7 7.18066C7.20625 7.18066 7.375 7.01191 7.375 6.80566V4.93066H9.25C9.45625 4.93066 9.625 4.76191 9.625 4.55566C9.625 4.34941 9.45625 4.18066 9.25 4.18066H7.375V2.30566C7.375 2.09941 7.20625 1.93066 7 1.93066C6.79375 1.93066 6.625 2.09941 6.625 2.30566V4.18066H4.75C4.54375 4.18066 4.375 4.34941 4.375 4.55566C4.375 4.76191 4.54375 4.93066 4.75 4.93066Z' fill='%23616161' stroke='%23616161' stroke-width='0.7'/%3e %3c/g%3e %3cpath class='jp-icon3' fill-rule='evenodd' clip-rule='evenodd' d='M11.5 9.5V11.5L2.5 11.5V9.5L11.5 9.5ZM12 8C12.5523 8 13 8.44772 13 9V12C13 12.5523 12.5523 13 12 13L2 13C1.44772 13 1 12.5523 1 12V9C1 8.44772 1.44771 8 2 8L12 8Z' fill='%23616161'/%3e %3cdefs%3e %3cclipPath id='clip0_137_19492'%3e %3crect class='jp-icon3' width='6' height='6' fill='white' transform='matrix(-1 0 0 1 10 1.55566)'/%3e %3c/clipPath%3e %3c/defs%3e %3c/svg%3e");
  --jp-icon-add-below: url("data:image/svg+xml,%3csvg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg clip-path='url(%23clip0_137_19498)'%3e %3cpath class='jp-icon3' d='M9.25 10.0693L7.375 10.0693L7.375 8.19434C7.375 7.98809 7.20625 7.81934 7 7.81934C6.79375 7.81934 6.625 7.98809 6.625 8.19434L6.625 10.0693L4.75 10.0693C4.54375 10.0693 4.375 10.2381 4.375 10.4443C4.375 10.6506 4.54375 10.8193 4.75 10.8193L6.625 10.8193L6.625 12.6943C6.625 12.9006 6.79375 13.0693 7 13.0693C7.20625 13.0693 7.375 12.9006 7.375 12.6943L7.375 10.8193L9.25 10.8193C9.45625 10.8193 9.625 10.6506 9.625 10.4443C9.625 10.2381 9.45625 10.0693 9.25 10.0693Z' fill='%23616161' stroke='%23616161' stroke-width='0.7'/%3e %3c/g%3e %3cpath class='jp-icon3' fill-rule='evenodd' clip-rule='evenodd' d='M2.5 5.5L2.5 3.5L11.5 3.5L11.5 5.5L2.5 5.5ZM2 7C1.44772 7 1 6.55228 1 6L1 3C1 2.44772 1.44772 2 2 2L12 2C12.5523 2 13 2.44772 13 3L13 6C13 6.55229 12.5523 7 12 7L2 7Z' fill='%23616161'/%3e %3cdefs%3e %3cclipPath id='clip0_137_19498'%3e %3crect class='jp-icon3' width='6' height='6' fill='white' transform='matrix(1 1.74846e-07 1.74846e-07 -1 4 13.4443)'/%3e %3c/clipPath%3e %3c/defs%3e %3c/svg%3e");
  --jp-icon-add: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-bell: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 16 16' version='1.1'%3e %3cpath class='jp-icon2 jp-icon-selectable' fill='%23333333' d='m8 0.29c-1.4 0-2.7 0.73-3.6 1.8-1.2 1.5-1.4 3.4-1.5 5.2-0.18 2.2-0.44 4-2.3 5.3l0.28 1.3h5c0.026 0.66 0.32 1.1 0.71 1.5 0.84 0.61 2 0.61 2.8 0 0.52-0.4 0.6-1 0.71-1.5h5l0.28-1.3c-1.9-0.97-2.2-3.3-2.3-5.3-0.13-1.8-0.26-3.7-1.5-5.2-0.85-1-2.2-1.8-3.6-1.8zm0 1.4c0.88 0 1.9 0.55 2.5 1.3 0.88 1.1 1.1 2.7 1.2 4.4 0.13 1.7 0.23 3.6 1.3 5.2h-10c1.1-1.6 1.2-3.4 1.3-5.2 0.13-1.7 0.3-3.3 1.2-4.4 0.59-0.72 1.6-1.3 2.5-1.3zm-0.74 12h1.5c-0.0015 0.28 0.015 0.79-0.74 0.79-0.73 0.0016-0.72-0.53-0.74-0.79z' /%3e %3c/svg%3e");
  --jp-icon-bug-dot: url("data:image/svg+xml,%3csvg width='24' height='24' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3 jp-icon-selectable' fill='%23616161'%3e %3cpath fill-rule='evenodd' clip-rule='evenodd' d='M17.19 8H20V10H17.91C17.96 10.33 18 10.66 18 11V12H20V14H18.5H18V14.0275C15.75 14.2762 14 16.1837 14 18.5C14 19.208 14.1635 19.8779 14.4549 20.4739C13.7063 20.8117 12.8757 21 12 21C9.78 21 7.85 19.79 6.81 18H4V16H6.09C6.04 15.67 6 15.34 6 15V14H4V12H6V11C6 10.66 6.04 10.33 6.09 10H4V8H6.81C7.26 7.22 7.88 6.55 8.62 6.04L7 4.41L8.41 3L10.59 5.17C11.04 5.06 11.51 5 12 5C12.49 5 12.96 5.06 13.42 5.17L15.59 3L17 4.41L15.37 6.04C16.12 6.55 16.74 7.22 17.19 8ZM10 16H14V14H10V16ZM10 12H14V10H10V12Z' fill='%23616161'/%3e %3cpath d='M22 18.5C22 20.433 20.433 22 18.5 22C16.567 22 15 20.433 15 18.5C15 16.567 16.567 15 18.5 15C20.433 15 22 16.567 22 18.5Z' fill='%23616161'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-bug: url("data:image/svg+xml,%3csvg viewBox='0 0 24 24' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3 jp-icon-selectable' fill='%23616161'%3e %3cpath d='M20 8h-2.81c-.45-.78-1.07-1.45-1.82-1.96L17 4.41 15.59 3l-2.17 2.17C12.96 5.06 12.49 5 12 5c-.49 0-.96.06-1.41.17L8.41 3 7 4.41l1.62 1.63C7.88 6.55 7.26 7.22 6.81 8H4v2h2.09c-.05.33-.09.66-.09 1v1H4v2h2v1c0 .34.04.67.09 1H4v2h2.81c1.04 1.79 2.97 3 5.19 3s4.15-1.21 5.19-3H20v-2h-2.09c.05-.33.09-.66.09-1v-1h2v-2h-2v-1c0-.34-.04-.67-.09-1H20V8zm-6 8h-4v-2h4v2zm0-4h-4v-2h4v2z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-build: url("data:image/svg+xml,%3csvg width='16' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M14.9 17.45C16.25 17.45 17.35 16.35 17.35 15C17.35 13.65 16.25 12.55 14.9 12.55C13.54 12.55 12.45 13.65 12.45 15C12.45 16.35 13.54 17.45 14.9 17.45ZM20.1 15.68L21.58 16.84C21.71 16.95 21.75 17.13 21.66 17.29L20.26 19.71C20.17 19.86 20 19.92 19.83 19.86L18.09 19.16C17.73 19.44 17.33 19.67 16.91 19.85L16.64 21.7C16.62 21.87 16.47 22 16.3 22H13.5C13.32 22 13.18 21.87 13.15 21.7L12.89 19.85C12.46 19.67 12.07 19.44 11.71 19.16L9.96002 19.86C9.81002 19.92 9.62002 19.86 9.54002 19.71L8.14002 17.29C8.05002 17.13 8.09002 16.95 8.22002 16.84L9.70002 15.68L9.65001 15L9.70002 14.31L8.22002 13.16C8.09002 13.05 8.05002 12.86 8.14002 12.71L9.54002 10.29C9.62002 10.13 9.81002 10.07 9.96002 10.13L11.71 10.84C12.07 10.56 12.46 10.32 12.89 10.15L13.15 8.28998C13.18 8.12998 13.32 7.99998 13.5 7.99998H16.3C16.47 7.99998 16.62 8.12998 16.64 8.28998L16.91 10.15C17.33 10.32 17.73 10.56 18.09 10.84L19.83 10.13C20 10.07 20.17 10.13 20.26 10.29L21.66 12.71C21.75 12.86 21.71 13.05 21.58 13.16L20.1 14.31L20.15 15L20.1 15.68Z'/%3e %3cpath d='M7.32966 7.44454C8.0831 7.00954 8.33932 6.05332 7.90432 5.29988C7.46932 4.54643 6.5081 4.28156 5.75466 4.71656C5.39176 4.92608 5.12695 5.27118 5.01849 5.67594C4.91004 6.08071 4.96682 6.51198 5.17634 6.87488C5.61134 7.62832 6.57622 7.87954 7.32966 7.44454ZM9.65718 4.79593L10.8672 4.95179C10.9628 4.97741 11.0402 5.07133 11.0382 5.18793L11.0388 6.98893C11.0455 7.10054 10.9616 7.19518 10.855 7.21054L9.66001 7.38083L9.23915 8.13188L9.66961 9.25745C9.70729 9.36271 9.66934 9.47699 9.57408 9.53199L8.01523 10.432C7.91131 10.492 7.79337 10.4677 7.72105 10.3824L6.98748 9.43188L6.10931 9.43083L5.34704 10.3905C5.28909 10.4702 5.17383 10.4905 5.07187 10.4339L3.51245 9.53293C3.41049 9.47633 3.37647 9.35741 3.41075 9.25679L3.86347 8.14093L3.61749 7.77488L3.42347 7.37883L2.23075 7.21297C2.12647 7.19235 2.04049 7.10342 2.04245 6.98682L2.04187 5.18582C2.04383 5.06922 2.11909 4.97958 2.21704 4.96922L3.42065 4.79393L3.86749 4.02788L3.41105 2.91731C3.37337 2.81204 3.41131 2.69776 3.51523 2.63776L5.07408 1.73776C5.16934 1.68276 5.28729 1.70704 5.35961 1.79231L6.11915 2.72788L6.98001 2.73893L7.72496 1.78922C7.79156 1.70458 7.91548 1.67922 8.00879 1.74082L9.56821 2.64182C9.67017 2.69842 9.71285 2.81234 9.68723 2.90797L9.21718 4.03383L9.46316 4.39988L9.65718 4.79593Z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-caret-down-empty-thin: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 20 20'%3e %3cg class='jp-icon3' fill='%23616161' shape-rendering='geometricPrecision'%3e %3cpolygon class='st1' points='9.9%2c13.6 3.6%2c7.4 4.4%2c6.6 9.9%2c12.2 15.4%2c6.7 16.1%2c7.4 '/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-caret-down-empty: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 18 18'%3e %3cg class='jp-icon3' fill='%23616161' shape-rendering='geometricPrecision'%3e %3cpath d='M5.2%2c5.9L9%2c9.7l3.8-3.8l1.2%2c1.2l-4.9%2c5l-4.9-5L5.2%2c5.9z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-caret-down: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 18 18'%3e %3cg class='jp-icon3' fill='%23616161' shape-rendering='geometricPrecision'%3e %3cpath d='M5.2%2c7.5L9%2c11.2l3.8-3.8H5.2z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-caret-left: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 18 18'%3e %3cg class='jp-icon3' fill='%23616161' shape-rendering='geometricPrecision'%3e %3cpath d='M10.8%2c12.8L7.1%2c9l3.8-3.8l0%2c7.6H10.8z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-caret-right: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 18 18'%3e %3cg class='jp-icon3' fill='%23616161' shape-rendering='geometricPrecision'%3e %3cpath d='M7.2%2c5.2L10.9%2c9l-3.8%2c3.8V5.2H7.2z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-caret-up-empty-thin: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 20 20'%3e %3cg class='jp-icon3' fill='%23616161' shape-rendering='geometricPrecision'%3e %3cpolygon class='st1' points='15.4%2c13.3 9.9%2c7.7 4.4%2c13.2 3.6%2c12.5 9.9%2c6.3 16.1%2c12.6 '/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-caret-up: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 18 18'%3e %3cg class='jp-icon3' fill='%23616161' shape-rendering='geometricPrecision'%3e %3cpath d='M5.2%2c10.5L9%2c6.8l3.8%2c3.8H5.2z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-case-sensitive: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 20 20'%3e %3cg class='jp-icon2' fill='%23414141'%3e %3crect x='2' y='2' width='16' height='16'/%3e %3c/g%3e %3cg class='jp-icon-accent2' fill='white'%3e %3cpath d='M7.6%2c8h0.9l3.5%2c8h-1.1L10%2c14H6l-0.9%2c2H4L7.6%2c8z M8%2c9.1L6.4%2c13h3.2L8%2c9.1z'/%3e %3cpath d='M16.6%2c9.8c-0.2%2c0.1-0.4%2c0.1-0.7%2c0.1c-0.2%2c0-0.4-0.1-0.6-0.2c-0.1-0.1-0.2-0.4-0.2-0.7 c-0.3%2c0.3-0.6%2c0.5-0.9%2c0.7c-0.3%2c0.1-0.7%2c0.2-1.1%2c0.2c-0.3%2c0-0.5%2c0-0.7-0.1c-0.2-0.1-0.4-0.2-0.6-0.3c-0.2-0.1-0.3-0.3-0.4-0.5 c-0.1-0.2-0.1-0.4-0.1-0.7c0-0.3%2c0.1-0.6%2c0.2-0.8c0.1-0.2%2c0.3-0.4%2c0.4-0.5C12%2c7%2c12.2%2c6.9%2c12.5%2c6.8c0.2-0.1%2c0.5-0.1%2c0.7-0.2 c0.3-0.1%2c0.5-0.1%2c0.7-0.1c0.2%2c0%2c0.4-0.1%2c0.6-0.1c0.2%2c0%2c0.3-0.1%2c0.4-0.2c0.1-0.1%2c0.2-0.2%2c0.2-0.4c0-1-1.1-1-1.3-1 c-0.4%2c0-1.4%2c0-1.4%2c1.2h-0.9c0-0.4%2c0.1-0.7%2c0.2-1c0.1-0.2%2c0.3-0.4%2c0.5-0.6c0.2-0.2%2c0.5-0.3%2c0.8-0.3C13.3%2c4%2c13.6%2c4%2c13.9%2c4 c0.3%2c0%2c0.5%2c0%2c0.8%2c0.1c0.3%2c0%2c0.5%2c0.1%2c0.7%2c0.2c0.2%2c0.1%2c0.4%2c0.3%2c0.5%2c0.5C16%2c5%2c16%2c5.2%2c16%2c5.6v2.9c0%2c0.2%2c0%2c0.4%2c0%2c0.5 c0%2c0.1%2c0.1%2c0.2%2c0.3%2c0.2c0.1%2c0%2c0.2%2c0%2c0.3%2c0V9.8z M15.2%2c6.9c-1.2%2c0.6-3.1%2c0.2-3.1%2c1.4c0%2c1.4%2c3.1%2c1%2c3.1-0.5V6.9z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-check: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3 jp-icon-selectable' fill='%23616161'%3e %3cpath d='M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-circle-empty: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M12 2C6.47 2 2 6.47 2 12s4.47 10 10 10 10-4.47 10-10S17.53 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-circle: url("data:image/svg+xml,%3csvg viewBox='0 0 18 18' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3ccircle cx='9' cy='9' r='8'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-clear: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cmask id='donutHole'%3e %3crect width='24' height='24' fill='white' /%3e %3ccircle cx='12' cy='12' r='8' fill='black'/%3e %3c/mask%3e %3cg class='jp-icon3' fill='%23616161'%3e %3crect height='18' width='2' x='11' y='3' transform='rotate(315%2c 12%2c 12)'/%3e %3ccircle cx='12' cy='12' r='10' mask='url(%23donutHole)'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-close: url("data:image/svg+xml,%3csvg viewBox='0 0 24 24' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon-none jp-icon-selectable-inverse jp-icon3-hover' fill='none'%3e %3ccircle cx='12' cy='12' r='11'/%3e %3c/g%3e %3cg class='jp-icon3 jp-icon-selectable jp-icon-accent2-hover' fill='%23616161'%3e %3cpath d='M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z'/%3e %3c/g%3e %3cg class='jp-icon-none jp-icon-busy' fill='none'%3e %3ccircle cx='12' cy='12' r='7'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-code-check: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='24' viewBox='0 0 24 24'%3e %3cg class='jp-icon3 jp-icon-selectable' fill='%23616161' shape-rendering='geometricPrecision'%3e %3cpath d='M6.59%2c3.41L2%2c8L6.59%2c12.6L8%2c11.18L4.82%2c8L8%2c4.82L6.59%2c3.41M12.41%2c3.41L11%2c4.82L14.18%2c8L11%2c11.18L12.41%2c12.6L17%2c8L12.41%2c3.41M21.59%2c11.59L13.5%2c19.68L9.83%2c16L8.42%2c17.41L13.5%2c22.5L23%2c13L21.59%2c11.59Z' /%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-code: url("data:image/svg+xml,%3csvg width='22' height='22' viewBox='0 0 28 28' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M11.4 18.6L6.8 14L11.4 9.4L10 8L4 14L10 20L11.4 18.6ZM16.6 18.6L21.2 14L16.6 9.4L18 8L24 14L18 20L16.6 18.6V18.6Z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-collapse-all: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M8 2c1 0 11 0 12 0s2 1 2 2c0 1 0 11 0 12s0 2-2 2C20 14 20 4 20 4S10 4 6 4c0-2 1-2 2-2z' /%3e %3cpath d='M18 8c0-1-1-2-2-2S5 6 4 6s-2 1-2 2c0 1 0 11 0 12s1 2 2 2c1 0 11 0 12 0s2-1 2-2c0-1 0-11 0-12zm-2 0v12H4V8z' /%3e %3cpath d='M6 13v2h8v-2z' /%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-console: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 200 200'%3e %3cg class='jp-console-icon-background-color jp-icon-selectable' fill='%230288D1'%3e %3cpath d='M20 19.8h160v159.9H20z'/%3e %3c/g%3e %3cg class='jp-console-icon-color jp-icon-selectable-inverse' fill='white'%3e %3cpath d='M105 127.3h40v12.8h-40zM51.1 77L74 99.9l-23.3 23.3 10.5 10.5 23.3-23.3L95 99.9 84.5 89.4 61.6 66.5z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-copy: url("data:image/svg+xml,%3csvg viewBox='0 0 18 18' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M11.9%2c1H3.2C2.4%2c1%2c1.7%2c1.7%2c1.7%2c2.5v10.2h1.5V2.5h8.7V1z M14.1%2c3.9h-8c-0.8%2c0-1.5%2c0.7-1.5%2c1.5v10.2c0%2c0.8%2c0.7%2c1.5%2c1.5%2c1.5h8 c0.8%2c0%2c1.5-0.7%2c1.5-1.5V5.4C15.5%2c4.6%2c14.9%2c3.9%2c14.1%2c3.9z M14.1%2c15.5h-8V5.4h8V15.5z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-copyright: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' enable-background='new 0 0 24 24' height='24' viewBox='0 0 24 24' width='24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M11.88%2c9.14c1.28%2c0.06%2c1.61%2c1.15%2c1.63%2c1.66h1.79c-0.08-1.98-1.49-3.19-3.45-3.19C9.64%2c7.61%2c8%2c9%2c8%2c12.14 c0%2c1.94%2c0.93%2c4.24%2c3.84%2c4.24c2.22%2c0%2c3.41-1.65%2c3.44-2.95h-1.79c-0.03%2c0.59-0.45%2c1.38-1.63%2c1.44C10.55%2c14.83%2c10%2c13.81%2c10%2c12.14 C10%2c9.25%2c11.28%2c9.16%2c11.88%2c9.14z M12%2c2C6.48%2c2%2c2%2c6.48%2c2%2c12s4.48%2c10%2c10%2c10s10-4.48%2c10-10S17.52%2c2%2c12%2c2z M12%2c20c-4.41%2c0-8-3.59-8-8 s3.59-8%2c8-8s8%2c3.59%2c8%2c8S16.41%2c20%2c12%2c20z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-cut: url("data:image/svg+xml,%3csvg viewBox='0 0 24 24' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M9.64 7.64c.23-.5.36-1.05.36-1.64 0-2.21-1.79-4-4-4S2 3.79 2 6s1.79 4 4 4c.59 0 1.14-.13 1.64-.36L10 12l-2.36 2.36C7.14 14.13 6.59 14 6 14c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4c0-.59-.13-1.14-.36-1.64L12 14l7 7h3v-1L9.64 7.64zM6 8c-1.1 0-2-.89-2-2s.9-2 2-2 2 .89 2 2-.9 2-2 2zm0 12c-1.1 0-2-.89-2-2s.9-2 2-2 2 .89 2 2-.9 2-2 2zm6-7.5c-.28 0-.5-.22-.5-.5s.22-.5.5-.5.5.22.5.5-.22.5-.5.5zM19 3l-6 6 2 2 7-7V3z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-delete: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='16px' height='16px'%3e %3cpath d='M0 0h24v24H0z' fill='none' /%3e %3cpath class='jp-icon3' fill='%23626262' d='M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z' /%3e %3c/svg%3e");
  --jp-icon-download: url("data:image/svg+xml,%3csvg viewBox='0 0 24 24' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-duplicate: url("data:image/svg+xml,%3csvg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath class='jp-icon3' fill-rule='evenodd' clip-rule='evenodd' d='M2.79998 0.875H8.89582C9.20061 0.875 9.44998 1.13914 9.44998 1.46198C9.44998 1.78482 9.20061 2.04896 8.89582 2.04896H3.35415C3.04936 2.04896 2.79998 2.3131 2.79998 2.63594V9.67969C2.79998 10.0025 2.55061 10.2667 2.24582 10.2667C1.94103 10.2667 1.69165 10.0025 1.69165 9.67969V2.04896C1.69165 1.40328 2.1904 0.875 2.79998 0.875ZM5.36665 11.9V4.55H11.0833V11.9H5.36665ZM4.14165 4.14167C4.14165 3.69063 4.50728 3.325 4.95832 3.325H11.4917C11.9427 3.325 12.3083 3.69063 12.3083 4.14167V12.3083C12.3083 12.7594 11.9427 13.125 11.4917 13.125H4.95832C4.50728 13.125 4.14165 12.7594 4.14165 12.3083V4.14167Z' fill='%23616161'/%3e %3cpath class='jp-icon3' d='M9.43574 8.26507H8.36431V9.3365C8.36431 9.45435 8.26788 9.55078 8.15002 9.55078C8.03217 9.55078 7.93574 9.45435 7.93574 9.3365V8.26507H6.86431C6.74645 8.26507 6.65002 8.16864 6.65002 8.05078C6.65002 7.93292 6.74645 7.8365 6.86431 7.8365H7.93574V6.76507C7.93574 6.64721 8.03217 6.55078 8.15002 6.55078C8.26788 6.55078 8.36431 6.64721 8.36431 6.76507V7.8365H9.43574C9.5536 7.8365 9.65002 7.93292 9.65002 8.05078C9.65002 8.16864 9.5536 8.26507 9.43574 8.26507Z' fill='%23616161' stroke='%23616161' stroke-width='0.5'/%3e %3c/svg%3e");
  --jp-icon-edit: url("data:image/svg+xml,%3csvg viewBox='0 0 24 24' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-ellipses: url("data:image/svg+xml,%3csvg viewBox='0 0 24 24' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3ccircle cx='5' cy='12' r='2'/%3e %3ccircle cx='12' cy='12' r='2'/%3e %3ccircle cx='19' cy='12' r='2'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-error: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e%3ccircle cx='12' cy='19' r='2'/%3e%3cpath d='M10 3h4v12h-4z'/%3e%3c/g%3e %3cpath fill='none' d='M0 0h24v24H0z'/%3e %3c/svg%3e");
  --jp-icon-expand-all: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M8 2c1 0 11 0 12 0s2 1 2 2c0 1 0 11 0 12s0 2-2 2C20 14 20 4 20 4S10 4 6 4c0-2 1-2 2-2z' /%3e %3cpath d='M18 8c0-1-1-2-2-2S5 6 4 6s-2 1-2 2c0 1 0 11 0 12s1 2 2 2c1 0 11 0 12 0s2-1 2-2c0-1 0-11 0-12zm-2 0v12H4V8z' /%3e %3cpath d='M11 10H9v3H6v2h3v3h2v-3h3v-2h-3z' /%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-extension: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M20.5 11H19V7c0-1.1-.9-2-2-2h-4V3.5C13 2.12 11.88 1 10.5 1S8 2.12 8 3.5V5H4c-1.1 0-1.99.9-1.99 2v3.8H3.5c1.49 0 2.7 1.21 2.7 2.7s-1.21 2.7-2.7 2.7H2V20c0 1.1.9 2 2 2h3.8v-1.5c0-1.49 1.21-2.7 2.7-2.7 1.49 0 2.7 1.21 2.7 2.7V22H17c1.1 0 2-.9 2-2v-4h1.5c1.38 0 2.5-1.12 2.5-2.5S21.88 11 20.5 11z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-fast-forward: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-file-upload: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M9 16h6v-6h4l-7-7-7 7h4zm-4 2h14v2H5z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-file: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 22 22'%3e %3cpath class='jp-icon3 jp-icon-selectable' fill='%23616161' d='M19.3 8.2l-5.5-5.5c-.3-.3-.7-.5-1.2-.5H3.9c-.8.1-1.6.9-1.6 1.8v14.1c0 .9.7 1.6 1.6 1.6h14.2c.9 0 1.6-.7 1.6-1.6V9.4c.1-.5-.1-.9-.4-1.2zm-5.8-3.3l3.4 3.6h-3.4V4.9zm3.9 12.7H4.7c-.1 0-.2 0-.2-.2V4.7c0-.2.1-.3.2-.3h7.2v4.4s0 .8.3 1.1c.3.3 1.1.3 1.1.3h4.3v7.2s-.1.2-.2.2z'/%3e %3c/svg%3e");
  --jp-icon-filter-dot: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='white'%3e %3cpath d='M14%2c12V19.88C14.04%2c20.18 13.94%2c20.5 13.71%2c20.71C13.32%2c21.1 12.69%2c21.1 12.3%2c20.71L10.29%2c18.7C10.06%2c18.47 9.96%2c18.16 10%2c17.87V12H9.97L4.21%2c4.62C3.87%2c4.19 3.95%2c3.56 4.38%2c3.22C4.57%2c3.08 4.78%2c3 5%2c3V3H19V3C19.22%2c3 19.43%2c3.08 19.62%2c3.22C20.05%2c3.56 20.13%2c4.19 19.79%2c4.62L14.03%2c12H14Z' /%3e %3c/g%3e %3cg class='jp-icon-dot' fill='white'%3e %3ccircle cx='18' cy='17' r='3'%3e%3c/circle%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-filter-list: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-filter: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='white'%3e %3cpath d='M14%2c12V19.88C14.04%2c20.18 13.94%2c20.5 13.71%2c20.71C13.32%2c21.1 12.69%2c21.1 12.3%2c20.71L10.29%2c18.7C10.06%2c18.47 9.96%2c18.16 10%2c17.87V12H9.97L4.21%2c4.62C3.87%2c4.19 3.95%2c3.56 4.38%2c3.22C4.57%2c3.08 4.78%2c3 5%2c3V3H19V3C19.22%2c3 19.43%2c3.08 19.62%2c3.22C20.05%2c3.56 20.13%2c4.19 19.79%2c4.62L14.03%2c12H14Z' /%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-folder-favorite: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' height='24px' viewBox='0 0 24 24' width='24px' fill='black'%3e %3cpath d='M0 0h24v24H0V0z' fill='none'/%3e%3cpath class='jp-icon3 jp-icon-selectable' fill='%23616161' d='M20 6h-8l-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm-2.06 11L15 15.28 12.06 17l.78-3.33-2.59-2.24 3.41-.29L15 8l1.34 3.14 3.41.29-2.59 2.24.78 3.33z'/%3e %3c/svg%3e");
  --jp-icon-folder: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cpath class='jp-icon3 jp-icon-selectable' fill='%23616161' d='M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z'/%3e %3c/svg%3e");
  --jp-icon-home: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' height='24px' viewBox='0 0 24 24' width='24px' fill='black'%3e %3cpath d='M0 0h24v24H0z' fill='none'/%3e%3cpath class='jp-icon3 jp-icon-selectable' fill='%23616161' d='M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z'/%3e %3c/svg%3e");
  --jp-icon-html5: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 512 512'%3e %3cpath class='jp-icon0 jp-icon-selectable' fill='black' d='M108.4 0h23v22.8h21.2V0h23v69h-23V46h-21v23h-23.2M206 23h-20.3V0h63.7v23H229v46h-23m53.5-69h24.1l14.8 24.3L313.2 0h24.1v69h-23V34.8l-16.1 24.8-16.1-24.8V69h-22.6m89.2-69h23v46.2h32.6V69h-55.6'/%3e %3cpath class='jp-icon-selectable' fill='%23e44d26' d='M107.6 471l-33-370.4h362.8l-33 370.2L255.7 512'/%3e %3cpath class='jp-icon-selectable' fill='%23f16529' d='M256 480.5V131h148.3L376 447'/%3e %3cpath class='jp-icon-selectable-inverse' fill='%23ebebeb' d='M142 176.3h114v45.4h-64.2l4.2 46.5h60v45.3H154.4m2 22.8H202l3.2 36.3 50.8 13.6v47.4l-93.2-26'/%3e %3cpath class='jp-icon-selectable-inverse' fill='white' d='M369.6 176.3H255.8v45.4h109.6m-4.1 46.5H255.8v45.4h56l-5.3 59-50.7 13.6v47.2l93-25.8'/%3e %3c/svg%3e");
  --jp-icon-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 22 22'%3e %3cpath class='jp-icon-brand4 jp-icon-selectable-inverse' fill='white' d='M2.2 2.2h17.5v17.5H2.2z'/%3e %3cpath class='jp-icon-brand0 jp-icon-selectable' fill='%233F51B5' d='M2.2 2.2v17.5h17.5l.1-17.5H2.2zm12.1 2.2c1.2 0 2.2 1 2.2 2.2s-1 2.2-2.2 2.2-2.2-1-2.2-2.2 1-2.2 2.2-2.2zM4.4 17.6l3.3-8.8 3.3 6.6 2.2-3.2 4.4 5.4H4.4z'/%3e %3c/svg%3e");
  --jp-icon-info: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 50.978 50.978'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M43.52%2c7.458C38.711%2c2.648%2c32.307%2c0%2c25.489%2c0C18.67%2c0%2c12.266%2c2.648%2c7.458%2c7.458 c-9.943%2c9.941-9.943%2c26.119%2c0%2c36.062c4.809%2c4.809%2c11.212%2c7.456%2c18.031%2c7.458c0%2c0%2c0.001%2c0%2c0.002%2c0 c6.816%2c0%2c13.221-2.648%2c18.029-7.458c4.809-4.809%2c7.457-11.212%2c7.457-18.03C50.977%2c18.67%2c48.328%2c12.266%2c43.52%2c7.458z M42.106%2c42.105c-4.432%2c4.431-10.332%2c6.872-16.615%2c6.872h-0.002c-6.285-0.001-12.187-2.441-16.617-6.872 c-9.162-9.163-9.162-24.071%2c0-33.233C13.303%2c4.44%2c19.204%2c2%2c25.489%2c2c6.284%2c0%2c12.186%2c2.44%2c16.617%2c6.872 c4.431%2c4.431%2c6.871%2c10.332%2c6.871%2c16.617C48.977%2c31.772%2c46.536%2c37.675%2c42.106%2c42.105z'/%3e %3cpath d='M23.578%2c32.218c-0.023-1.734%2c0.143-3.059%2c0.496-3.972c0.353-0.913%2c1.11-1.997%2c2.272-3.253 c0.468-0.536%2c0.923-1.062%2c1.367-1.575c0.626-0.753%2c1.104-1.478%2c1.436-2.175c0.331-0.707%2c0.495-1.541%2c0.495-2.5 c0-1.096-0.26-2.088-0.779-2.979c-0.565-0.879-1.501-1.336-2.806-1.369c-1.802%2c0.057-2.985%2c0.667-3.55%2c1.832 c-0.301%2c0.535-0.503%2c1.141-0.607%2c1.814c-0.139%2c0.707-0.207%2c1.432-0.207%2c2.174h-2.937c-0.091-2.208%2c0.407-4.114%2c1.493-5.719 c1.062-1.64%2c2.855-2.481%2c5.378-2.527c2.16%2c0.023%2c3.874%2c0.608%2c5.141%2c1.758c1.278%2c1.16%2c1.929%2c2.764%2c1.95%2c4.811 c0%2c1.142-0.137%2c2.111-0.41%2c2.911c-0.309%2c0.845-0.731%2c1.593-1.268%2c2.243c-0.492%2c0.65-1.068%2c1.318-1.73%2c2.002 c-0.65%2c0.697-1.313%2c1.479-1.987%2c2.346c-0.239%2c0.377-0.429%2c0.777-0.565%2c1.199c-0.16%2c0.959-0.217%2c1.951-0.171%2c2.979 C26.589%2c32.218%2c23.578%2c32.218%2c23.578%2c32.218z M23.578%2c38.22v-3.484h3.076v3.484H23.578z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-inspector: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cpath class='jp-inspector-icon-color jp-icon-selectable' fill='%23616161' d='M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm-5 14H4v-4h11v4zm0-5H4V9h11v4zm5 5h-4V9h4v9z'/%3e %3c/svg%3e");
  --jp-icon-json: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 22 22'%3e %3cg class='jp-json-icon-color jp-icon-selectable' fill='%23F9A825'%3e %3cpath d='M20.2 11.8c-1.6 0-1.7.5-1.7 1 0 .4.1.9.1 1.3.1.5.1.9.1 1.3 0 1.7-1.4 2.3-3.5 2.3h-.9v-1.9h.5c1.1 0 1.4 0 1.4-.8 0-.3 0-.6-.1-1 0-.4-.1-.8-.1-1.2 0-1.3 0-1.8 1.3-2-1.3-.2-1.3-.7-1.3-2 0-.4.1-.8.1-1.2.1-.4.1-.7.1-1 0-.8-.4-.7-1.4-.8h-.5V4.1h.9c2.2 0 3.5.7 3.5 2.3 0 .4-.1.9-.1 1.3-.1.5-.1.9-.1 1.3 0 .5.2 1 1.7 1v1.8zM1.8 10.1c1.6 0 1.7-.5 1.7-1 0-.4-.1-.9-.1-1.3-.1-.5-.1-.9-.1-1.3 0-1.6 1.4-2.3 3.5-2.3h.9v1.9h-.5c-1 0-1.4 0-1.4.8 0 .3 0 .6.1 1 0 .2.1.6.1 1 0 1.3 0 1.8-1.3 2C6 11.2 6 11.7 6 13c0 .4-.1.8-.1 1.2-.1.3-.1.7-.1 1 0 .8.3.8 1.4.8h.5v1.9h-.9c-2.1 0-3.5-.6-3.5-2.3 0-.4.1-.9.1-1.3.1-.5.1-.9.1-1.3 0-.5-.2-1-1.7-1v-1.9z'/%3e %3ccircle cx='11' cy='13.8' r='2.1'/%3e %3ccircle cx='11' cy='8.2' r='2.1'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-julia: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 325 300'%3e %3cg class='jp-brand0 jp-icon-selectable' fill='%23cb3c33'%3e %3cpath d='M 150.898438 225 C 150.898438 266.421875 117.320312 300 75.898438 300 C 34.476562 300 0.898438 266.421875 0.898438 225 C 0.898438 183.578125 34.476562 150 75.898438 150 C 117.320312 150 150.898438 183.578125 150.898438 225'/%3e %3c/g%3e %3cg class='jp-brand0 jp-icon-selectable' fill='%23389826'%3e %3cpath d='M 237.5 75 C 237.5 116.421875 203.921875 150 162.5 150 C 121.078125 150 87.5 116.421875 87.5 75 C 87.5 33.578125 121.078125 0 162.5 0 C 203.921875 0 237.5 33.578125 237.5 75'/%3e %3c/g%3e %3cg class='jp-brand0 jp-icon-selectable' fill='%239558b2'%3e %3cpath d='M 324.101562 225 C 324.101562 266.421875 290.523438 300 249.101562 300 C 207.679688 300 174.101562 266.421875 174.101562 225 C 174.101562 183.578125 207.679688 150 249.101562 150 C 290.523438 150 324.101562 183.578125 324.101562 225'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-jupyter-favicon: url("data:image/svg+xml,%3csvg width='152' height='165' viewBox='0 0 152 165' version='1.1' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-jupyter-icon-color' fill='%23F37726'%3e %3cpath transform='translate(0.078947%2c 110.582927)' d='M75.9422842%2c29.5804561 C43.3023947%2c29.5804561 14.7967832%2c17.6534634 0%2c0 C5.51083211%2c15.8406829 15.7815389%2c29.5667732 29.3904947%2c39.2784171 C42.9997%2c48.9898537 59.2737%2c54.2067805 75.9605789%2c54.2067805 C92.6474579%2c54.2067805 108.921458%2c48.9898537 122.530663%2c39.2784171 C136.139453%2c29.5667732 146.410284%2c15.8406829 151.921158%2c0 C137.087868%2c17.6534634 108.582589%2c29.5804561 75.9422842%2c29.5804561 L75.9422842%2c29.5804561 Z' /%3e %3cpath transform='translate(0.037368%2c 0.704878)' d='M75.9784579%2c24.6264073 C108.618763%2c24.6264073 137.124458%2c36.5534415 151.921158%2c54.2067805 C146.410284%2c38.366222 136.139453%2c24.6401317 122.530663%2c14.9284878 C108.921458%2c5.2168439 92.6474579%2c0 75.9605789%2c0 C59.2737%2c0 42.9997%2c5.2168439 29.3904947%2c14.9284878 C15.7815389%2c24.6401317 5.51083211%2c38.366222 0%2c54.2067805 C14.8330816%2c36.5899293 43.3385684%2c24.6264073 75.9784579%2c24.6264073 L75.9784579%2c24.6264073 Z' /%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-jupyter: url("data:image/svg+xml,%3csvg width='39' height='51' viewBox='0 0 39 51' xmlns='http://www.w3.org/2000/svg'%3e %3cg transform='translate(-1638 -2281)'%3e %3cg class='jp-jupyter-icon-color' fill='%23F37726'%3e %3cpath transform='translate(1639.74 2311.98)' d='M 18.2646 7.13411C 10.4145 7.13411 3.55872 4.2576 0 0C 1.32539 3.8204 3.79556 7.13081 7.0686 9.47303C 10.3417 11.8152 14.2557 13.0734 18.269 13.0734C 22.2823 13.0734 26.1963 11.8152 29.4694 9.47303C 32.7424 7.13081 35.2126 3.8204 36.538 0C 32.9705 4.2576 26.1148 7.13411 18.2646 7.13411Z'/%3e %3cpath transform='translate(1639.73 2285.48)' d='M 18.2733 5.93931C 26.1235 5.93931 32.9793 8.81583 36.538 13.0734C 35.2126 9.25303 32.7424 5.94262 29.4694 3.6004C 26.1963 1.25818 22.2823 0 18.269 0C 14.2557 0 10.3417 1.25818 7.0686 3.6004C 3.79556 5.94262 1.32539 9.25303 0 13.0734C 3.56745 8.82463 10.4232 5.93931 18.2733 5.93931Z'/%3e %3c/g%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath transform='translate(1669.3 2281.31)' d='M 5.89353 2.844C 5.91889 3.43165 5.77085 4.01367 5.46815 4.51645C 5.16545 5.01922 4.72168 5.42015 4.19299 5.66851C 3.6643 5.91688 3.07444 6.00151 2.49805 5.91171C 1.92166 5.8219 1.38463 5.5617 0.954898 5.16401C 0.52517 4.76633 0.222056 4.24903 0.0839037 3.67757C -0.0542483 3.10611 -0.02123 2.50617 0.178781 1.95364C 0.378793 1.4011 0.736809 0.920817 1.20754 0.573538C 1.67826 0.226259 2.24055 0.0275919 2.82326 0.00267229C 3.60389 -0.0307115 4.36573 0.249789 4.94142 0.782551C 5.51711 1.31531 5.85956 2.05676 5.89353 2.844Z'/%3e %3cpath transform='translate(1639.8 2323.81)' d='M 7.42789 3.58338C 7.46008 4.3243 7.27355 5.05819 6.89193 5.69213C 6.51031 6.32607 5.95075 6.83156 5.28411 7.1446C 4.61747 7.45763 3.87371 7.56414 3.14702 7.45063C 2.42032 7.33712 1.74336 7.0087 1.20184 6.50695C 0.660328 6.0052 0.27861 5.35268 0.105017 4.63202C -0.0685757 3.91135 -0.0262361 3.15494 0.226675 2.45856C 0.479587 1.76217 0.931697 1.15713 1.52576 0.720033C 2.11983 0.282935 2.82914 0.0334395 3.56389 0.00313344C 4.54667 -0.0374033 5.50529 0.316706 6.22961 0.987835C 6.95393 1.65896 7.38484 2.59235 7.42789 3.58338L 7.42789 3.58338Z'/%3e %3cpath transform='translate(1638.36 2286.06)' d='M 2.27471 4.39629C 1.84363 4.41508 1.41671 4.30445 1.04799 4.07843C 0.679268 3.8524 0.385328 3.52114 0.203371 3.12656C 0.0214136 2.73198 -0.0403798 2.29183 0.0258116 1.86181C 0.0920031 1.4318 0.283204 1.03126 0.575213 0.710883C 0.867222 0.39051 1.24691 0.164708 1.66622 0.0620592C 2.08553 -0.0405897 2.52561 -0.0154714 2.93076 0.134235C 3.33591 0.283941 3.68792 0.551505 3.94222 0.90306C 4.19652 1.25462 4.34169 1.67436 4.35935 2.10916C 4.38299 2.69107 4.17678 3.25869 3.78597 3.68746C 3.39516 4.11624 2.85166 4.37116 2.27471 4.39629L 2.27471 4.39629Z'/%3e %3c/g%3e %3c/g%3e%3e %3c/svg%3e");
  --jp-icon-jupyterlab-wordmark: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='200' viewBox='0 0 1860.8 475'%3e %3cg class='jp-icon2' fill='%234E4E4E' transform='translate(480.136401%2c 64.271493)'%3e %3cg transform='translate(0.000000%2c 58.875566)'%3e %3cg transform='translate(0.087603%2c 0.140294)'%3e %3cpath d='M-426.9%2c169.8c0%2c48.7-3.7%2c64.7-13.6%2c76.4c-10.8%2c10-25%2c15.5-39.7%2c15.5l3.7%2c29 c22.8%2c0.3%2c44.8-7.9%2c61.9-23.1c17.8-18.5%2c24-44.1%2c24-83.3V0H-427v170.1L-426.9%2c169.8L-426.9%2c169.8z'/%3e %3c/g%3e %3c/g%3e %3cg transform='translate(155.045296%2c 56.837104)'%3e %3cg transform='translate(1.562453%2c 1.799842)'%3e %3cpath d='M-312%2c148c0%2c21%2c0%2c39.5%2c1.7%2c55.4h-31.8l-2.1-33.3h-0.8c-6.7%2c11.6-16.4%2c21.3-28%2c27.9 c-11.6%2c6.6-24.8%2c10-38.2%2c9.8c-31.4%2c0-69-17.7-69-89V0h36.4v112.7c0%2c38.7%2c11.6%2c64.7%2c44.6%2c64.7c10.3-0.2%2c20.4-3.5%2c28.9-9.4 c8.5-5.9%2c15.1-14.3%2c18.9-23.9c2.2-6.1%2c3.3-12.5%2c3.3-18.9V0.2h36.4V148H-312L-312%2c148z'/%3e %3c/g%3e %3c/g%3e %3cg transform='translate(390.013322%2c 53.479638)'%3e %3cg transform='translate(1.706458%2c 0.231425)'%3e %3cpath d='M-478.6%2c71.4c0-26-0.8-47-1.7-66.7h32.7l1.7%2c34.8h0.8c7.1-12.5%2c17.5-22.8%2c30.1-29.7 c12.5-7%2c26.7-10.3%2c41-9.8c48.3%2c0%2c84.7%2c41.7%2c84.7%2c103.3c0%2c73.1-43.7%2c109.2-91%2c109.2c-12.1%2c0.5-24.2-2.2-35-7.8 c-10.8-5.6-19.9-13.9-26.6-24.2h-0.8V291h-36v-220L-478.6%2c71.4L-478.6%2c71.4z M-442.6%2c125.6c0.1%2c5.1%2c0.6%2c10.1%2c1.7%2c15.1 c3%2c12.3%2c9.9%2c23.3%2c19.8%2c31.1c9.9%2c7.8%2c22.1%2c12.1%2c34.7%2c12.1c38.5%2c0%2c60.7-31.9%2c60.7-78.5c0-40.7-21.1-75.6-59.5-75.6 c-12.9%2c0.4-25.3%2c5.1-35.3%2c13.4c-9.9%2c8.3-16.9%2c19.7-19.6%2c32.4c-1.5%2c4.9-2.3%2c10-2.5%2c15.1V125.6L-442.6%2c125.6L-442.6%2c125.6z'/%3e %3c/g%3e %3c/g%3e %3cg transform='translate(606.740726%2c 56.837104)'%3e %3cg transform='translate(0.751226%2c 1.989299)'%3e %3cpath d='M-440.8%2c0l43.7%2c120.1c4.5%2c13.4%2c9.5%2c29.4%2c12.8%2c41.7h0.8c3.7-12.2%2c7.9-27.7%2c12.8-42.4 l39.7-119.2h38.5L-346.9%2c145c-26%2c69.7-43.7%2c105.4-68.6%2c127.2c-12.5%2c11.7-27.9%2c20-44.6%2c23.9l-9.1-31.1 c11.7-3.9%2c22.5-10.1%2c31.8-18.1c13.2-11.1%2c23.7-25.2%2c30.6-41.2c1.5-2.8%2c2.5-5.7%2c2.9-8.8c-0.3-3.3-1.2-6.6-2.5-9.7L-480.2%2c0.1 h39.7L-440.8%2c0L-440.8%2c0z'/%3e %3c/g%3e %3c/g%3e %3cg transform='translate(822.748104%2c 0.000000)'%3e %3cg transform='translate(1.464050%2c 0.378914)'%3e %3cpath d='M-413.7%2c0v58.3h52v28.2h-52V196c0%2c25%2c7%2c39.5%2c27.3%2c39.5c7.1%2c0.1%2c14.2-0.7%2c21.1-2.5 l1.7%2c27.7c-10.3%2c3.7-21.3%2c5.4-32.2%2c5c-7.3%2c0.4-14.6-0.7-21.3-3.4c-6.8-2.7-12.9-6.8-17.9-12.1c-10.3-10.9-14.1-29-14.1-52.9 V86.5h-31V58.3h31V9.6L-413.7%2c0L-413.7%2c0z'/%3e %3c/g%3e %3c/g%3e %3cg transform='translate(974.433286%2c 53.479638)'%3e %3cg transform='translate(0.990034%2c 0.610339)'%3e %3cpath d='M-445.8%2c113c0.8%2c50%2c32.2%2c70.6%2c68.6%2c70.6c19%2c0.6%2c37.9-3%2c55.3-10.5l6.2%2c26.4 c-20.9%2c8.9-43.5%2c13.1-66.2%2c12.6c-61.5%2c0-98.3-41.2-98.3-102.5C-480.2%2c48.2-444.7%2c0-386.5%2c0c65.2%2c0%2c82.7%2c58.3%2c82.7%2c95.7 c-0.1%2c5.8-0.5%2c11.5-1.2%2c17.2h-140.6H-445.8L-445.8%2c113z M-339.2%2c86.6c0.4-23.5-9.5-60.1-50.4-60.1 c-36.8%2c0-52.8%2c34.4-55.7%2c60.1H-339.2L-339.2%2c86.6L-339.2%2c86.6z'/%3e %3c/g%3e %3c/g%3e %3cg transform='translate(1201.961058%2c 53.479638)'%3e %3cg transform='translate(1.179640%2c 0.705068)'%3e %3cpath d='M-478.6%2c68c0-23.9-0.4-44.5-1.7-63.4h31.8l1.2%2c39.9h1.7c9.1-27.3%2c31-44.5%2c55.3-44.5 c3.5-0.1%2c7%2c0.4%2c10.3%2c1.2v34.8c-4.1-0.9-8.2-1.3-12.4-1.2c-25.6%2c0-43.7%2c19.7-48.7%2c47.4c-1%2c5.7-1.6%2c11.5-1.7%2c17.2v108.3h-36V68 L-478.6%2c68z'/%3e %3c/g%3e %3c/g%3e %3c/g%3e %3cg class='jp-icon-warn0' fill='%23F37726'%3e %3cpath d='M1352.3%2c326.2h37V28h-37V326.2z M1604.8%2c326.2c-2.5-13.9-3.4-31.1-3.4-48.7v-76 c0-40.7-15.1-83.1-77.3-83.1c-25.6%2c0-50%2c7.1-66.8%2c18.1l8.4%2c24.4c14.3-9.2%2c34-15.1%2c53-15.1c41.6%2c0%2c46.2%2c30.2%2c46.2%2c47v4.2 c-78.6-0.4-122.3%2c26.5-122.3%2c75.6c0%2c29.4%2c21%2c58.4%2c62.2%2c58.4c29%2c0%2c50.9-14.3%2c62.2-30.2h1.3l2.9%2c25.6H1604.8z M1565.7%2c257.7 c0%2c3.8-0.8%2c8-2.1%2c11.8c-5.9%2c17.2-22.7%2c34-49.2%2c34c-18.9%2c0-34.9-11.3-34.9-35.3c0-39.5%2c45.8-46.6%2c86.2-45.8V257.7z M1698.5%2c326.2 l1.7-33.6h1.3c15.1%2c26.9%2c38.7%2c38.2%2c68.1%2c38.2c45.4%2c0%2c91.2-36.1%2c91.2-108.8c0.4-61.7-35.3-103.7-85.7-103.7 c-32.8%2c0-56.3%2c14.7-69.3%2c37.4h-0.8V28h-36.6v245.7c0%2c18.1-0.8%2c38.6-1.7%2c52.5H1698.5z M1704.8%2c208.2c0-5.9%2c1.3-10.9%2c2.1-15.1 c7.6-28.1%2c31.1-45.4%2c56.3-45.4c39.5%2c0%2c60.5%2c34.9%2c60.5%2c75.6c0%2c46.6-23.1%2c78.1-61.8%2c78.1c-26.9%2c0-48.3-17.6-55.5-43.3 c-0.8-4.2-1.7-8.8-1.7-13.4V208.2z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-kernel: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cpath class='jp-icon2' fill='%23616161' d='M15 9H9v6h6V9zm-2 4h-2v-2h2v2zm8-2V9h-2V7c0-1.1-.9-2-2-2h-2V3h-2v2h-2V3H9v2H7c-1.1 0-2 .9-2 2v2H3v2h2v2H3v2h2v2c0 1.1.9 2 2 2h2v2h2v-2h2v2h2v-2h2c1.1 0 2-.9 2-2v-2h2v-2h-2v-2h2zm-4 6H7V7h10v10z'/%3e %3c/svg%3e");
  --jp-icon-keyboard: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cpath class='jp-icon3 jp-icon-selectable' fill='%23616161' d='M20 5H4c-1.1 0-1.99.9-1.99 2L2 17c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm-9 3h2v2h-2V8zm0 3h2v2h-2v-2zM8 8h2v2H8V8zm0 3h2v2H8v-2zm-1 2H5v-2h2v2zm0-3H5V8h2v2zm9 7H8v-2h8v2zm0-4h-2v-2h2v2zm0-3h-2V8h2v2zm3 3h-2v-2h2v2zm0-3h-2V8h2v2z'/%3e %3c/svg%3e");
  --jp-icon-launch: url("data:image/svg+xml,%3csvg viewBox='0 0 32 32' width='32' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3 jp-icon-selectable' fill='%23616161'%3e %3cpath d='M26%2c28H6a2.0027%2c2.0027%2c0%2c0%2c1-2-2V6A2.0027%2c2.0027%2c0%2c0%2c1%2c6%2c4H16V6H6V26H26V16h2V26A2.0027%2c2.0027%2c0%2c0%2c1%2c26%2c28Z'/%3e %3cpolygon points='20 2 20 4 26.586 4 18 12.586 19.414 14 28 5.414 28 12 30 12 30 2 20 2'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-launcher: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cpath class='jp-icon3 jp-icon-selectable' fill='%23616161' d='M19 19H5V5h7V3H5a2 2 0 00-2 2v14a2 2 0 002 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z'/%3e %3c/svg%3e");
  --jp-icon-line-form: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cpath fill='white' d='M5.88 4.12L13.76 12l-7.88 7.88L8 22l10-10L8 2z'/%3e %3c/svg%3e");
  --jp-icon-link: url("data:image/svg+xml,%3csvg viewBox='0 0 24 24' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M3.9 12c0-1.71 1.39-3.1 3.1-3.1h4V7H7c-2.76 0-5 2.24-5 5s2.24 5 5 5h4v-1.9H7c-1.71 0-3.1-1.39-3.1-3.1zM8 13h8v-2H8v2zm9-6h-4v1.9h4c1.71 0 3.1 1.39 3.1 3.1s-1.39 3.1-3.1 3.1h-4V17h4c2.76 0 5-2.24 5-5s-2.24-5-5-5z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-list: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cpath class='jp-icon2 jp-icon-selectable' fill='%23616161' d='M19 5v14H5V5h14m1.1-2H3.9c-.5 0-.9.4-.9.9v16.2c0 .4.4.9.9.9h16.2c.4 0 .9-.5.9-.9V3.9c0-.5-.5-.9-.9-.9zM11 7h6v2h-6V7zm0 4h6v2h-6v-2zm0 4h6v2h-6zM7 7h2v2H7zm0 4h2v2H7zm0 4h2v2H7z'/%3e %3c/svg%3e");
  --jp-icon-markdown: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 22 22'%3e %3cpath class='jp-icon-contrast0 jp-icon-selectable' fill='%237B1FA2' d='M5 14.9h12l-6.1 6zm9.4-6.8c0-1.3-.1-2.9-.1-4.5-.4 1.4-.9 2.9-1.3 4.3l-1.3 4.3h-2L8.5 7.9c-.4-1.3-.7-2.9-1-4.3-.1 1.6-.1 3.2-.2 4.6L7 12.4H4.8l.7-11h3.3L10 5c.4 1.2.7 2.7 1 3.9.3-1.2.7-2.6 1-3.9l1.2-3.7h3.3l.6 11h-2.4l-.3-4.2z'/%3e %3c/svg%3e");
  --jp-icon-move-down: url("data:image/svg+xml,%3csvg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath class='jp-icon3' d='M12.471 7.52899C12.7632 7.23684 12.7632 6.76316 12.471 6.47101V6.47101C12.179 6.17905 11.7057 6.17884 11.4135 6.47054L7.75 10.1275V1.75C7.75 1.33579 7.41421 1 7 1V1C6.58579 1 6.25 1.33579 6.25 1.75V10.1275L2.59726 6.46822C2.30338 6.17381 1.82641 6.17359 1.53226 6.46774V6.46774C1.2383 6.7617 1.2383 7.2383 1.53226 7.53226L6.29289 12.2929C6.68342 12.6834 7.31658 12.6834 7.70711 12.2929L12.471 7.52899Z' fill='%23616161'/%3e %3c/svg%3e");
  --jp-icon-move-up: url("data:image/svg+xml,%3csvg width='14' height='14' viewBox='0 0 14 14' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath class='jp-icon3' d='M1.52899 6.47101C1.23684 6.76316 1.23684 7.23684 1.52899 7.52899V7.52899C1.82095 7.82095 2.29426 7.82116 2.58649 7.52946L6.25 3.8725V12.25C6.25 12.6642 6.58579 13 7 13V13C7.41421 13 7.75 12.6642 7.75 12.25V3.8725L11.4027 7.53178C11.6966 7.82619 12.1736 7.82641 12.4677 7.53226V7.53226C12.7617 7.2383 12.7617 6.7617 12.4677 6.46774L7.70711 1.70711C7.31658 1.31658 6.68342 1.31658 6.29289 1.70711L1.52899 6.47101Z' fill='%23616161'/%3e %3c/svg%3e");
  --jp-icon-new-folder: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M20 6h-8l-2-2H4c-1.11 0-1.99.89-1.99 2L2 18c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V8c0-1.11-.89-2-2-2zm-1 8h-3v3h-2v-3h-3v-2h3V9h2v3h3v2z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-not-trusted: url("data:image/svg+xml,%3csvg fill='none' xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 25 25'%3e %3cpath class='jp-icon2' stroke='%23333333' stroke-width='2' transform='translate(3 3)' d='M1.86094 11.4409C0.826448 8.77027 0.863779 6.05764 1.24907 4.19932C2.48206 3.93347 4.08068 3.40347 5.60102 2.8449C7.23549 2.2444 8.85666 1.5815 9.9876 1.09539C11.0597 1.58341 12.6094 2.2444 14.218 2.84339C15.7503 3.41394 17.3995 3.95258 18.7539 4.21385C19.1364 6.07177 19.1709 8.77722 18.139 11.4409C17.0303 14.3032 14.6668 17.1844 9.99999 18.9354C5.33319 17.1844 2.96968 14.3032 1.86094 11.4409Z'/%3e %3cpath class='jp-icon2' stroke='%23333333' stroke-width='2' transform='translate(9.31592 9.32031)' d='M7.36842 0L0 7.36479'/%3e %3cpath class='jp-icon2' stroke='%23333333' stroke-width='2' transform='translate(9.31592 16.6836) scale(1 -1)' d='M7.36842 0L0 7.36479'/%3e %3c/svg%3e");
  --jp-icon-notebook: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 22 22'%3e %3cg class='jp-notebook-icon-color jp-icon-selectable' fill='%23EF6C00'%3e %3cpath d='M18.7 3.3v15.4H3.3V3.3h15.4m1.5-1.5H1.8v18.3h18.3l.1-18.3z'/%3e %3cpath d='M16.5 16.5l-5.4-4.3-5.6 4.3v-11h11z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-numbering: url("data:image/svg+xml,%3csvg width='22' height='22' viewBox='0 0 28 28' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M4 19H6V19.5H5V20.5H6V21H4V22H7V18H4V19ZM5 10H6V6H4V7H5V10ZM4 13H5.8L4 15.1V16H7V15H5.2L7 12.9V12H4V13ZM9 7V9H23V7H9ZM9 21H23V19H9V21ZM9 15H23V13H9V15Z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-offline-bolt: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' width='16'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M12 2.02c-5.51 0-9.98 4.47-9.98 9.98s4.47 9.98 9.98 9.98 9.98-4.47 9.98-9.98S17.51 2.02 12 2.02zM11.48 20v-6.26H8L13 4v6.26h3.35L11.48 20z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-palette: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M18 13V20H4V6H9.02C9.07 5.29 9.24 4.62 9.5 4H4C2.9 4 2 4.9 2 6V20C2 21.1 2.9 22 4 22H18C19.1 22 20 21.1 20 20V15L18 13ZM19.3 8.89C19.74 8.19 20 7.38 20 6.5C20 4.01 17.99 2 15.5 2C13.01 2 11 4.01 11 6.5C11 8.99 13.01 11 15.49 11C16.37 11 17.19 10.74 17.88 10.3L21 13.42L22.42 12L19.3 8.89ZM15.5 9C14.12 9 13 7.88 13 6.5C13 5.12 14.12 4 15.5 4C16.88 4 18 5.12 18 6.5C18 7.88 16.88 9 15.5 9Z'/%3e %3cpath fill-rule='evenodd' clip-rule='evenodd' d='M4 6H9.01894C9.00639 6.16502 9 6.33176 9 6.5C9 8.81577 10.211 10.8487 12.0343 12H9V14H16V12.9811C16.5703 12.9377 17.12 12.8207 17.6396 12.6396L18 13V20H4V6ZM8 8H6V10H8V8ZM6 12H8V14H6V12ZM8 16H6V18H8V16ZM9 16H16V18H9V16Z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-paste: url("data:image/svg+xml,%3csvg height='24' viewBox='0 0 24 24' width='24' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M19 2h-4.18C14.4.84 13.3 0 12 0c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm7 18H5V4h2v3h10V4h2v16z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-pdf: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 22 22' width='16'%3e %3cpath transform='rotate(45)' class='jp-icon-selectable' fill='%23FF2A2A' d='m 22.344369%2c-3.0163642 h 5.638604 v 1.5792433 h -3.549227 v 1.50869299 h 3.337576 V 1.6508154 h -3.337576 v 3.4352613 h -2.089377 z m -7.136444%2c1.5792433 v 4.9439543 h 0.74892 q 1.280761%2c0 1.953703%2c-0.6349535 0.678369%2c-0.6349535 0.678369%2c-1.8451641 0%2c-1.20478355 -0.672942%2c-1.83431011 -0.672942%2c-0.62952659 -1.95913%2c-0.62952659 z m -2.089377%2c-1.5792433 h 2.203343 q 1.845164%2c0 2.746039%2c0.2659207 0.906301%2c0.2604937 1.552108%2c0.8900203 0.56983%2c0.5481223 0.846605%2c1.26448006 0.276774%2c0.71635781 0.276774%2c1.62265894 0%2c0.9171551 -0.276774%2c1.6389399 -0.276775%2c0.7163578 -0.846605%2c1.26448 -0.651234%2c0.6295266 -1.562962%2c0.8954473 -0.911728%2c0.2604937 -2.735185%2c0.2604937 h -2.203343 z m -8.1458565%2c0 h 3.467823 q 1.5466816%2c0 2.3715785%2c0.689223 0.830324%2c0.6837961 0.830324%2c1.95370314 0%2c1.27533397 -0.830324%2c1.96455706 Q 9.9871961%2c2.274915 8.4405145%2c2.274915 H 7.0620684 V 5.0860767 H 4.9726915 Z m 2.0893769%2c1.5141199 v 2.26303943 h 1.155941 q 0.6078188%2c0 0.9388629%2c-0.29305547 0.3310441%2c-0.29848241 0.3310441%2c-0.84117772 0%2c-0.54269531 -0.3310441%2c-0.83575074 -0.3310441%2c-0.2930555 -0.9388629%2c-0.2930555 z' /%3e %3c/svg%3e");
  --jp-icon-python: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='-10 -10 131.16136169433594 132.38899993896484'%3e %3cpath class='jp-icon-selectable' fill='%23306998' d='M 54.918785%2c9.1927421e-4 C 50.335132%2c0.02221727 45.957846%2c0.41313697 42.106285%2c1.0946693 30.760069%2c3.0991731 28.700036%2c7.2947714 28.700035%2c15.032169 v 10.21875 h 26.8125 v 3.40625 h -26.8125 -10.0625 c -7.792459%2c0 -14.6157588%2c4.683717 -16.7499998%2c13.59375 -2.46181998%2c10.212966 -2.57101508%2c16.586023 0%2c27.25 1.9059283%2c7.937852 6.4575432%2c13.593748 14.2499998%2c13.59375 h 9.21875 v -12.25 c 0%2c-8.849902 7.657144%2c-16.656248 16.75%2c-16.65625 h 26.78125 c 7.454951%2c0 13.406253%2c-6.138164 13.40625%2c-13.625 v -25.53125 c 0%2c-7.2663386 -6.12998%2c-12.7247771 -13.40625%2c-13.9374997 C 64.281548%2c0.32794397 59.502438%2c-0.02037903 54.918785%2c9.1927421e-4 Z m -14.5%2c8.21875012579 c 2.769547%2c0 5.03125%2c2.2986456 5.03125%2c5.1249996 -2e-6%2c2.816336 -2.261703%2c5.09375 -5.03125%2c5.09375 -2.779476%2c-1e-6 -5.03125%2c-2.277415 -5.03125%2c-5.09375 -10e-7%2c-2.826353 2.251774%2c-5.1249996 5.03125%2c-5.1249996 z'/%3e %3cpath class='jp-icon-selectable' fill='%23ffd43b' d='m 85.637535%2c28.657169 v 11.90625 c 0%2c9.230755 -7.825895%2c16.999999 -16.75%2c17 h -26.78125 c -7.335833%2c0 -13.406249%2c6.278483 -13.40625%2c13.625 v 25.531247 c 0%2c7.266344 6.318588%2c11.540324 13.40625%2c13.625004 8.487331%2c2.49561 16.626237%2c2.94663 26.78125%2c0 6.750155%2c-1.95439 13.406253%2c-5.88761 13.40625%2c-13.625004 V 86.500919 h -26.78125 v -3.40625 h 26.78125 13.406254 c 7.792461%2c0 10.696251%2c-5.435408 13.406241%2c-13.59375 2.79933%2c-8.398886 2.68022%2c-16.475776 0%2c-27.25 -1.92578%2c-7.757441 -5.60387%2c-13.59375 -13.406241%2c-13.59375 z m -15.0625%2c64.65625 c 2.779478%2c3e-6 5.03125%2c2.277417 5.03125%2c5.093747 -2e-6%2c2.826354 -2.251775%2c5.125004 -5.03125%2c5.125004 -2.76955%2c0 -5.03125%2c-2.29865 -5.03125%2c-5.125004 2e-6%2c-2.81633 2.261697%2c-5.093747 5.03125%2c-5.093747 z'/%3e %3c/svg%3e");
  --jp-icon-r-kernel: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 22 22'%3e %3cpath class='jp-icon-contrast3 jp-icon-selectable' fill='%232196F3' d='M4.4 2.5c1.2-.1 2.9-.3 4.9-.3 2.5 0 4.1.4 5.2 1.3 1 .7 1.5 1.9 1.5 3.5 0 2-1.4 3.5-2.9 4.1 1.2.4 1.7 1.6 2.2 3 .6 1.9 1 3.9 1.3 4.6h-3.8c-.3-.4-.8-1.7-1.2-3.7s-1.2-2.6-2.6-2.6h-.9v6.4H4.4V2.5zm3.7 6.9h1.4c1.9 0 2.9-.9 2.9-2.3s-1-2.3-2.8-2.3c-.7 0-1.3 0-1.6.2v4.5h.1v-.1z'/%3e %3c/svg%3e");
  --jp-icon-react: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='150 150 541.9 295.3'%3e %3cg class='jp-icon-brand2 jp-icon-selectable' fill='%2361DAFB'%3e %3cpath d='M666.3 296.5c0-32.5-40.7-63.3-103.1-82.4 14.4-63.6 8-114.2-20.2-130.4-6.5-3.8-14.1-5.6-22.4-5.6v22.3c4.6 0 8.3.9 11.4 2.6 13.6 7.8 19.5 37.5 14.9 75.7-1.1 9.4-2.9 19.3-5.1 29.4-19.6-4.8-41-8.5-63.5-10.9-13.5-18.5-27.5-35.3-41.6-50 32.6-30.3 63.2-46.9 84-46.9V78c-27.5 0-63.5 19.6-99.9 53.6-36.4-33.8-72.4-53.2-99.9-53.2v22.3c20.7 0 51.4 16.5 84 46.6-14 14.7-28 31.4-41.3 49.9-22.6 2.4-44 6.1-63.6 11-2.3-10-4-19.7-5.2-29-4.7-38.2 1.1-67.9 14.6-75.8 3-1.8 6.9-2.6 11.5-2.6V78.5c-8.4 0-16 1.8-22.6 5.6-28.1 16.2-34.4 66.7-19.9 130.1-62.2 19.2-102.7 49.9-102.7 82.3 0 32.5 40.7 63.3 103.1 82.4-14.4 63.6-8 114.2 20.2 130.4 6.5 3.8 14.1 5.6 22.5 5.6 27.5 0 63.5-19.6 99.9-53.6 36.4 33.8 72.4 53.2 99.9 53.2 8.4 0 16-1.8 22.6-5.6 28.1-16.2 34.4-66.7 19.9-130.1 62-19.1 102.5-49.9 102.5-82.3zm-130.2-66.7c-3.7 12.9-8.3 26.2-13.5 39.5-4.1-8-8.4-16-13.1-24-4.6-8-9.5-15.8-14.4-23.4 14.2 2.1 27.9 4.7 41 7.9zm-45.8 106.5c-7.8 13.5-15.8 26.3-24.1 38.2-14.9 1.3-30 2-45.2 2-15.1 0-30.2-.7-45-1.9-8.3-11.9-16.4-24.6-24.2-38-7.6-13.1-14.5-26.4-20.8-39.8 6.2-13.4 13.2-26.8 20.7-39.9 7.8-13.5 15.8-26.3 24.1-38.2 14.9-1.3 30-2 45.2-2 15.1 0 30.2.7 45 1.9 8.3 11.9 16.4 24.6 24.2 38 7.6 13.1 14.5 26.4 20.8 39.8-6.3 13.4-13.2 26.8-20.7 39.9zm32.3-13c5.4 13.4 10 26.8 13.8 39.8-13.1 3.2-26.9 5.9-41.2 8 4.9-7.7 9.8-15.6 14.4-23.7 4.6-8 8.9-16.1 13-24.1zM421.2 430c-9.3-9.6-18.6-20.3-27.8-32 9 .4 18.2.7 27.5.7 9.4 0 18.7-.2 27.8-.7-9 11.7-18.3 22.4-27.5 32zm-74.4-58.9c-14.2-2.1-27.9-4.7-41-7.9 3.7-12.9 8.3-26.2 13.5-39.5 4.1 8 8.4 16 13.1 24 4.7 8 9.5 15.8 14.4 23.4zM420.7 163c9.3 9.6 18.6 20.3 27.8 32-9-.4-18.2-.7-27.5-.7-9.4 0-18.7.2-27.8.7 9-11.7 18.3-22.4 27.5-32zm-74 58.9c-4.9 7.7-9.8 15.6-14.4 23.7-4.6 8-8.9 16-13 24-5.4-13.4-10-26.8-13.8-39.8 13.1-3.1 26.9-5.8 41.2-7.9zm-90.5 125.2c-35.4-15.1-58.3-34.9-58.3-50.6 0-15.7 22.9-35.6 58.3-50.6 8.6-3.7 18-7 27.7-10.1 5.7 19.6 13.2 40 22.5 60.9-9.2 20.8-16.6 41.1-22.2 60.6-9.9-3.1-19.3-6.5-28-10.2zM310 490c-13.6-7.8-19.5-37.5-14.9-75.7 1.1-9.4 2.9-19.3 5.1-29.4 19.6 4.8 41 8.5 63.5 10.9 13.5 18.5 27.5 35.3 41.6 50-32.6 30.3-63.2 46.9-84 46.9-4.5-.1-8.3-1-11.3-2.7zm237.2-76.2c4.7 38.2-1.1 67.9-14.6 75.8-3 1.8-6.9 2.6-11.5 2.6-20.7 0-51.4-16.5-84-46.6 14-14.7 28-31.4 41.3-49.9 22.6-2.4 44-6.1 63.6-11 2.3 10.1 4.1 19.8 5.2 29.1zm38.5-66.7c-8.6 3.7-18 7-27.7 10.1-5.7-19.6-13.2-40-22.5-60.9 9.2-20.8 16.6-41.1 22.2-60.6 9.9 3.1 19.3 6.5 28.1 10.2 35.4 15.1 58.3 34.9 58.3 50.6-.1 15.7-23 35.6-58.4 50.6zM320.8 78.4z'/%3e %3ccircle cx='420.9' cy='296.5' r='45.7'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-redo: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' height='24' viewBox='0 0 24 24' width='16'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M0 0h24v24H0z' fill='none'/%3e%3cpath d='M18.4 10.6C16.55 8.99 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-refresh: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 18 18'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M9 13.5c-2.49 0-4.5-2.01-4.5-4.5S6.51 4.5 9 4.5c1.24 0 2.36.52 3.17 1.33L10 8h5V3l-1.76 1.76C12.15 3.68 10.66 3 9 3 5.69 3 3.01 5.69 3.01 9S5.69 15 9 15c2.97 0 5.43-2.16 5.9-5h-1.52c-.46 2-2.24 3.5-4.38 3.5z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-regex: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 20 20'%3e %3cg class='jp-icon2' fill='%23414141'%3e %3crect x='2' y='2' width='16' height='16'/%3e %3c/g%3e %3cg class='jp-icon-accent2' fill='white'%3e %3ccircle class='st2' cx='5.5' cy='14.5' r='1.5'/%3e %3crect x='12' y='4' class='st2' width='1' height='8'/%3e %3crect x='8.5' y='7.5' transform='matrix(0.866 -0.5 0.5 0.866 -2.3255 7.3219)' class='st2' width='8' height='1'/%3e %3crect x='12' y='4' transform='matrix(0.5 -0.866 0.866 0.5 -0.6779 14.8252)' class='st2' width='1' height='8'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-run: url("data:image/svg+xml,%3csvg height='24' viewBox='0 0 24 24' width='24' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M8 5v14l11-7z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-running: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 512 512'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M256 8C119 8 8 119 8 256s111 248 248 248 248-111 248-248S393 8 256 8zm96 328c0 8.8-7.2 16-16 16H176c-8.8 0-16-7.2-16-16V176c0-8.8 7.2-16 16-16h160c8.8 0 16 7.2 16 16v160z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-save: url("data:image/svg+xml,%3csvg height='24' viewBox='0 0 24 24' width='24' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M17 3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V7l-4-4zm-5 16c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3zm3-10H5V5h10v4z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-search: url("data:image/svg+xml,%3csvg viewBox='0 0 18 18' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M12.1%2c10.9h-0.7l-0.2-0.2c0.8-0.9%2c1.3-2.2%2c1.3-3.5c0-3-2.4-5.4-5.4-5.4S1.8%2c4.2%2c1.8%2c7.1s2.4%2c5.4%2c5.4%2c5.4 c1.3%2c0%2c2.5-0.5%2c3.5-1.3l0.2%2c0.2v0.7l4.1%2c4.1l1.2-1.2L12.1%2c10.9z M7.1%2c10.9c-2.1%2c0-3.7-1.7-3.7-3.7s1.7-3.7%2c3.7-3.7s3.7%2c1.7%2c3.7%2c3.7 S9.2%2c10.9%2c7.1%2c10.9z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-settings: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cpath class='jp-icon3 jp-icon-selectable' fill='%23616161' d='M19.43 12.98c.04-.32.07-.64.07-.98s-.03-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.3-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65A.488.488 0 0014 2h-4c-.25 0-.46.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69.98l-2.49-1c-.23-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64l2.11 1.65c-.04.32-.07.65-.07.98s.03.66.07.98l-2.11 1.65c-.19.15-.24.42-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.03.24.24.42.49.42h4c.25 0 .46-.18.49-.42l.38-2.65c.61-.25 1.17-.59 1.69-.98l2.49 1c.23.09.49 0 .61-.22l2-3.46c.12-.22.07-.49-.12-.64l-2.11-1.65zM12 15.5c-1.93 0-3.5-1.57-3.5-3.5s1.57-3.5 3.5-3.5 3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5z'/%3e %3c/svg%3e");
  --jp-icon-share: url("data:image/svg+xml,%3csvg width='16' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M 18 2 C 16.35499 2 15 3.3549904 15 5 C 15 5.1909529 15.021791 5.3771224 15.056641 5.5585938 L 7.921875 9.7207031 C 7.3985399 9.2778539 6.7320771 9 6 9 C 4.3549904 9 3 10.35499 3 12 C 3 13.64501 4.3549904 15 6 15 C 6.7320771 15 7.3985399 14.722146 7.921875 14.279297 L 15.056641 18.439453 C 15.021555 18.621514 15 18.808386 15 19 C 15 20.64501 16.35499 22 18 22 C 19.64501 22 21 20.64501 21 19 C 21 17.35499 19.64501 16 18 16 C 17.26748 16 16.601593 16.279328 16.078125 16.722656 L 8.9433594 12.558594 C 8.9782095 12.377122 9 12.190953 9 12 C 9 11.809047 8.9782095 11.622878 8.9433594 11.441406 L 16.078125 7.2792969 C 16.60146 7.7221461 17.267923 8 18 8 C 19.64501 8 21 6.6450096 21 5 C 21 3.3549904 19.64501 2 18 2 z M 18 4 C 18.564129 4 19 4.4358706 19 5 C 19 5.5641294 18.564129 6 18 6 C 17.435871 6 17 5.5641294 17 5 C 17 4.4358706 17.435871 4 18 4 z M 6 11 C 6.5641294 11 7 11.435871 7 12 C 7 12.564129 6.5641294 13 6 13 C 5.4358706 13 5 12.564129 5 12 C 5 11.435871 5.4358706 11 6 11 z M 18 18 C 18.564129 18 19 18.435871 19 19 C 19 19.564129 18.564129 20 18 20 C 17.435871 20 17 19.564129 17 19 C 17 18.435871 17.435871 18 18 18 z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-spreadsheet: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 22 22'%3e %3cpath class='jp-icon-contrast1 jp-icon-selectable' fill='%234CAF50' d='M2.2 2.2v17.6h17.6V2.2H2.2zm15.4 7.7h-5.5V4.4h5.5v5.5zM9.9 4.4v5.5H4.4V4.4h5.5zm-5.5 7.7h5.5v5.5H4.4v-5.5zm7.7 5.5v-5.5h5.5v5.5h-5.5z'/%3e %3c/svg%3e");
  --jp-icon-stop: url("data:image/svg+xml,%3csvg height='24' viewBox='0 0 24 24' width='24' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M0 0h24v24H0z' fill='none'/%3e %3cpath d='M6 6h12v12H6z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-tab: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h10v4h8v10z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-table-rows: url("data:image/svg+xml,%3csvg height='24' viewBox='0 0 24 24' width='24' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M0 0h24v24H0z' fill='none'/%3e %3cpath d='M21%2c8H3V4h18V8z M21%2c10H3v4h18V10z M21%2c16H3v4h18V16z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-tag: url("data:image/svg+xml,%3csvg width='28' height='28' viewBox='0 0 43 28' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M28.8332 12.334L32.9998 16.5007L37.1665 12.334H28.8332Z'/%3e %3cpath d='M16.2095 21.6104C15.6873 22.1299 14.8443 22.1299 14.3248 21.6104L6.9829 14.7245C6.5724 14.3394 6.08313 13.6098 6.04786 13.0482C5.95347 11.5288 6.02002 8.61944 6.06621 7.07695C6.08281 6.51477 6.55548 6.04347 7.11804 6.03055C9.08863 5.98473 13.2638 5.93579 13.6518 6.32425L21.7369 13.639C22.256 14.1585 21.7851 15.4724 21.262 15.9946L16.2095 21.6104ZM9.77585 8.265C9.33551 7.82566 8.62351 7.82566 8.1828 8.265C7.74346 8.70571 7.74346 9.41733 8.1828 9.85667C8.62382 10.2964 9.33582 10.2964 9.77585 9.85667C10.2156 9.41733 10.2156 8.70533 9.77585 8.265Z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-terminal: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24' %3e %3crect class='jp-terminal-icon-background-color jp-icon-selectable' width='20' height='20' transform='translate(2 2)' fill='%23333333'/%3e %3cpath class='jp-terminal-icon-color jp-icon-selectable-inverse' d='M5.05664 8.76172C5.05664 8.59766 5.03125 8.45312 4.98047 8.32812C4.93359 8.19922 4.85547 8.08203 4.74609 7.97656C4.64062 7.87109 4.5 7.77539 4.32422 7.68945C4.15234 7.59961 3.94336 7.51172 3.69727 7.42578C3.30273 7.28516 2.94336 7.13672 2.61914 6.98047C2.29492 6.82422 2.01758 6.64258 1.78711 6.43555C1.56055 6.22852 1.38477 5.98828 1.25977 5.71484C1.13477 5.4375 1.07227 5.10938 1.07227 4.73047C1.07227 4.39844 1.12891 4.0957 1.24219 3.82227C1.35547 3.54492 1.51562 3.30469 1.72266 3.10156C1.92969 2.89844 2.17969 2.73437 2.47266 2.60938C2.76562 2.48438 3.0918 2.4043 3.45117 2.36914V1.10938H4.38867V2.38086C4.74023 2.42773 5.05664 2.52344 5.33789 2.66797C5.61914 2.8125 5.85742 3.00195 6.05273 3.23633C6.25195 3.4668 6.4043 3.74023 6.50977 4.05664C6.61914 4.36914 6.67383 4.7207 6.67383 5.11133H5.04492C5.04492 4.63867 4.9375 4.28125 4.72266 4.03906C4.50781 3.79297 4.2168 3.66992 3.84961 3.66992C3.65039 3.66992 3.47656 3.69727 3.32812 3.75195C3.18359 3.80273 3.06445 3.87695 2.9707 3.97461C2.87695 4.06836 2.80664 4.17969 2.75977 4.30859C2.7168 4.4375 2.69531 4.57812 2.69531 4.73047C2.69531 4.88281 2.7168 5.01953 2.75977 5.14062C2.80664 5.25781 2.88281 5.36719 2.98828 5.46875C3.09766 5.57031 3.24023 5.66797 3.41602 5.76172C3.5918 5.85156 3.81055 5.94336 4.07227 6.03711C4.4668 6.18555 4.82422 6.33984 5.14453 6.5C5.46484 6.65625 5.73828 6.83984 5.96484 7.05078C6.19531 7.25781 6.37109 7.5 6.49219 7.77734C6.61719 8.05078 6.67969 8.375 6.67969 8.75C6.67969 9.09375 6.62305 9.4043 6.50977 9.68164C6.39648 9.95508 6.23438 10.1914 6.02344 10.3906C5.8125 10.5898 5.55859 10.75 5.26172 10.8711C4.96484 10.9883 4.63281 11.0645 4.26562 11.0996V12.248H3.33398V11.0996C3.00195 11.0684 2.67969 10.9961 2.36719 10.8828C2.05469 10.7656 1.77734 10.5977 1.53516 10.3789C1.29688 10.1602 1.10547 9.88477 0.960938 9.55273C0.816406 9.2168 0.744141 8.81445 0.744141 8.3457H2.37891C2.37891 8.62695 2.41992 8.86328 2.50195 9.05469C2.58398 9.24219 2.68945 9.39258 2.81836 9.50586C2.95117 9.61523 3.10156 9.69336 3.26953 9.74023C3.4375 9.78711 3.60938 9.81055 3.78516 9.81055C4.20312 9.81055 4.51953 9.71289 4.73438 9.51758C4.94922 9.32227 5.05664 9.07031 5.05664 8.76172ZM13.418 12.2715H8.07422V11H13.418V12.2715Z' transform='translate(3.95264 6)' fill='white'/%3e %3c/svg%3e");
  --jp-icon-text-editor: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 24'%3e %3cpath class='jp-text-editor-icon-color jp-icon-selectable' fill='%23616161' d='M15 15H3v2h12v-2zm0-8H3v2h12V7zM3 13h18v-2H3v2zm0 8h18v-2H3v2zM3 3v2h18V3H3z'/%3e %3c/svg%3e");
  --jp-icon-toc: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24'%3e %3cg class='jp-icon3 jp-icon-selectable' fill='%23616161'%3e %3cpath d='M7%2c5H21V7H7V5M7%2c13V11H21V13H7M4%2c4.5A1.5%2c1.5 0 0%2c1 5.5%2c6A1.5%2c1.5 0 0%2c1 4%2c7.5A1.5%2c1.5 0 0%2c1 2.5%2c6A1.5%2c1.5 0 0%2c1 4%2c4.5M4%2c10.5A1.5%2c1.5 0 0%2c1 5.5%2c12A1.5%2c1.5 0 0%2c1 4%2c13.5A1.5%2c1.5 0 0%2c1 2.5%2c12A1.5%2c1.5 0 0%2c1 4%2c10.5M7%2c19V17H21V19H7M4%2c16.5A1.5%2c1.5 0 0%2c1 5.5%2c18A1.5%2c1.5 0 0%2c1 4%2c19.5A1.5%2c1.5 0 0%2c1 2.5%2c18A1.5%2c1.5 0 0%2c1 4%2c16.5Z' /%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-tree-view: url("data:image/svg+xml,%3csvg height='24' viewBox='0 0 24 24' width='24' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M0 0h24v24H0z' fill='none'/%3e %3cpath d='M22 11V3h-7v3H9V3H2v8h7V8h2v10h4v3h7v-8h-7v3h-2V8h2v3z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-trusted: url("data:image/svg+xml,%3csvg fill='none' xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 24 25'%3e %3cpath class='jp-icon2' stroke='%23333333' stroke-width='2' transform='translate(2 3)' d='M1.86094 11.4409C0.826448 8.77027 0.863779 6.05764 1.24907 4.19932C2.48206 3.93347 4.08068 3.40347 5.60102 2.8449C7.23549 2.2444 8.85666 1.5815 9.9876 1.09539C11.0597 1.58341 12.6094 2.2444 14.218 2.84339C15.7503 3.41394 17.3995 3.95258 18.7539 4.21385C19.1364 6.07177 19.1709 8.77722 18.139 11.4409C17.0303 14.3032 14.6668 17.1844 9.99999 18.9354C5.3332 17.1844 2.96968 14.3032 1.86094 11.4409Z'/%3e %3cpath class='jp-icon2' fill='%23333333' stroke='%23333333' transform='translate(8 9.86719)' d='M2.86015 4.86535L0.726549 2.99959L0 3.63045L2.86015 6.13157L8 0.630872L7.27857 0L2.86015 4.86535Z'/%3e %3c/svg%3e");
  --jp-icon-undo: url("data:image/svg+xml,%3csvg viewBox='0 0 24 24' width='16' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-user: url("data:image/svg+xml,%3csvg width='16' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' fill='%23616161'%3e %3cpath d='M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-users: url("data:image/svg+xml,%3csvg width='24' height='24' version='1.1' viewBox='0 0 36 24' xmlns='http://www.w3.org/2000/svg'%3e %3cg class='jp-icon3' transform='matrix(1.7327 0 0 1.7327 -3.6282 .099577)' fill='%23616161'%3e %3cpath transform='matrix(1.5%2c0%2c0%2c1.5%2c0%2c-6)' d='m12.186 7.5098c-1.0535 0-1.9757 0.5665-2.4785 1.4102 0.75061 0.31277 1.3974 0.82648 1.873 1.4727h3.4863c0-1.592-1.2889-2.8828-2.8809-2.8828z'/%3e %3cpath d='m20.465 2.3895a2.1885 2.1885 0 0 1-2.1884 2.1885 2.1885 2.1885 0 0 1-2.1885-2.1885 2.1885 2.1885 0 0 1 2.1885-2.1885 2.1885 2.1885 0 0 1 2.1884 2.1885z'/%3e %3cpath transform='matrix(1.5%2c0%2c0%2c1.5%2c0%2c-6)' d='m3.5898 8.4219c-1.1126 0-2.0137 0.90111-2.0137 2.0137h2.8145c0.26797-0.37309 0.5907-0.70435 0.95898-0.97852-0.34433-0.61688-1.0031-1.0352-1.7598-1.0352z'/%3e %3cpath d='m6.9154 4.623a1.5294 1.5294 0 0 1-1.5294 1.5294 1.5294 1.5294 0 0 1-1.5294-1.5294 1.5294 1.5294 0 0 1 1.5294-1.5294 1.5294 1.5294 0 0 1 1.5294 1.5294z'/%3e %3cpath d='m6.135 13.535c0-3.2392 2.6259-5.865 5.865-5.865 3.2392 0 5.865 2.6259 5.865 5.865z'/%3e %3ccircle cx='12' cy='3.7685' r='2.9685'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-vega: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 22 22'%3e %3cg class='jp-icon1 jp-icon-selectable' fill='%23212121'%3e %3cpath d='M10.6 5.4l2.2-3.2H2.2v7.3l4-6.6z'/%3e %3cpath d='M15.8 2.2l-4.4 6.6L7 6.3l-4.8 8v5.5h17.6V2.2h-4zm-7 15.4H5.5v-4.4h3.3v4.4zm4.4 0H9.8V9.8h3.4v7.8zm4.4 0h-3.4V6.5h3.4v11.1z'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-word: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 20 20'%3e %3cg class='jp-icon2' fill='%23414141'%3e %3crect x='2' y='2' width='16' height='16'/%3e %3c/g%3e %3cg class='jp-icon-accent2' transform='translate(.43 .0401)' fill='white'%3e %3cpath d='m4.14 8.76q0.0682-1.89 2.42-1.89 1.16 0 1.68 0.42 0.567 0.41 0.567 1.16v3.47q0 0.462 0.514 0.462 0.103 0 0.2-0.0231v0.714q-0.399 0.103-0.651 0.103-0.452 0-0.693-0.22-0.231-0.2-0.284-0.662-0.956 0.872-2 0.872-0.903 0-1.47-0.472-0.525-0.472-0.525-1.26 0-0.262 0.0452-0.472 0.0567-0.22 0.116-0.378 0.0682-0.168 0.231-0.304 0.158-0.147 0.262-0.242 0.116-0.0914 0.368-0.168 0.262-0.0914 0.399-0.126 0.136-0.0452 0.472-0.103 0.336-0.0578 0.504-0.0798 0.158-0.0231 0.567-0.0798 0.556-0.0682 0.777-0.221 0.22-0.152 0.22-0.441v-0.252q0-0.43-0.357-0.662-0.336-0.231-0.976-0.231-0.662 0-0.998 0.262-0.336 0.252-0.399 0.798zm1.89 3.68q0.788 0 1.26-0.41 0.504-0.42 0.504-0.903v-1.05q-0.284 0.136-0.861 0.231-0.567 0.0914-0.987 0.158-0.42 0.0682-0.766 0.326-0.336 0.252-0.336 0.704t0.304 0.704 0.861 0.252z' stroke-width='1.05'/%3e %3cpath d='m10 4.56h0.945v3.15q0.651-0.976 1.89-0.976 1.16 0 1.89 0.84 0.682 0.84 0.682 2.31 0 1.47-0.704 2.42-0.704 0.882-1.89 0.882-1.26 0-1.89-1.02v0.766h-0.85zm2.62 3.04q-0.746 0-1.16 0.64-0.452 0.63-0.452 1.68 0 1.05 0.452 1.68t1.16 0.63q0.777 0 1.26-0.63 0.494-0.64 0.494-1.68 0-1.05-0.472-1.68-0.462-0.64-1.26-0.64z' stroke-width='1.05'/%3e %3cpath d='m2.73 15.8 13.6 0.0081c0.0069 0 0-2.6 0-2.6 0-0.0078-1.15 0-1.15 0-0.0069 0-0.0083 1.5-0.0083 1.5-2e-3 -0.0014-11.3-0.0014-11.3-0.0014l-0.00592-1.5c0-0.0078-1.17 0.0013-1.17 0.0013z' stroke-width='.975'/%3e %3c/g%3e %3c/svg%3e");
  --jp-icon-yaml: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='16' viewBox='0 0 22 22'%3e %3cg class='jp-icon-contrast2 jp-icon-selectable' fill='%23D81B60'%3e %3cpath d='M7.2 18.6v-5.4L3 5.6h3.3l1.4 3.1c.3.9.6 1.6 1 2.5.3-.8.6-1.6 1-2.5l1.4-3.1h3.4l-4.4 7.6v5.5l-2.9-.1z'/%3e %3ccircle class='st0' cx='17.6' cy='16.5' r='2.1'/%3e %3ccircle class='st0' cx='17.6' cy='11' r='2.1'/%3e %3c/g%3e %3c/svg%3e");
}

/* Icon CSS class declarations */

.jp-AddAboveIcon {
  background-image: var(--jp-icon-add-above);
}

.jp-AddBelowIcon {
  background-image: var(--jp-icon-add-below);
}

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}

.jp-BellIcon {
  background-image: var(--jp-icon-bell);
}

.jp-BugDotIcon {
  background-image: var(--jp-icon-bug-dot);
}

.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}

.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}

.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}

.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}

.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}

.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}

.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}

.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}

.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}

.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}

.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}

.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}

.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}

.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}

.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}

.jp-CodeCheckIcon {
  background-image: var(--jp-icon-code-check);
}

.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}

.jp-CollapseAllIcon {
  background-image: var(--jp-icon-collapse-all);
}

.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}

.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}

.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
}

.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}

.jp-DeleteIcon {
  background-image: var(--jp-icon-delete);
}

.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}

.jp-DuplicateIcon {
  background-image: var(--jp-icon-duplicate);
}

.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}

.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}

.jp-ErrorIcon {
  background-image: var(--jp-icon-error);
}

.jp-ExpandAllIcon {
  background-image: var(--jp-icon-expand-all);
}

.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}

.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}

.jp-FileIcon {
  background-image: var(--jp-icon-file);
}

.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}

.jp-FilterDotIcon {
  background-image: var(--jp-icon-filter-dot);
}

.jp-FilterIcon {
  background-image: var(--jp-icon-filter);
}

.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}

.jp-FolderFavoriteIcon {
  background-image: var(--jp-icon-folder-favorite);
}

.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}

.jp-HomeIcon {
  background-image: var(--jp-icon-home);
}

.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}

.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}

.jp-InfoIcon {
  background-image: var(--jp-icon-info);
}

.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}

.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}

.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
}

.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}

.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}

.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}

.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}

.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}

.jp-LaunchIcon {
  background-image: var(--jp-icon-launch);
}

.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}

.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}

.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}

.jp-ListIcon {
  background-image: var(--jp-icon-list);
}

.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}

.jp-MoveDownIcon {
  background-image: var(--jp-icon-move-down);
}

.jp-MoveUpIcon {
  background-image: var(--jp-icon-move-up);
}

.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}

.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}

.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}

.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}

.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}

.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}

.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}

.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}

.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}

.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}

.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}

.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}

.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}

.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}

.jp-RunIcon {
  background-image: var(--jp-icon-run);
}

.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}

.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}

.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}

.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}

.jp-ShareIcon {
  background-image: var(--jp-icon-share);
}

.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}

.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}

.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}

.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}

.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}

.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}

.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}

.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}

.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}

.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}

.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}

.jp-UserIcon {
  background-image: var(--jp-icon-user);
}

.jp-UsersIcon {
  background-image: var(--jp-icon-users);
}

.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}

.jp-WordIcon {
  background-image: var(--jp-icon-word);
}

.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.lm-TabBar .lm-TabBar-addButton {
  align-items: center;
  display: flex;
  padding: 4px;
  padding-bottom: 5px;
  margin-right: 1px;
  background-color: var(--jp-layout-color2);
}

.lm-TabBar .lm-TabBar-addButton:hover {
  background-color: var(--jp-layout-color1);
}

.lm-DockPanel-tabBar .lm-TabBar-tab {
  width: var(--jp-private-horizontal-tab-width);
}

.lm-DockPanel-tabBar .lm-TabBar-content {
  flex: unset;
}

.lm-DockPanel-tabBar[data-orientation='horizontal'] {
  flex: 1 1 auto;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}

/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}

.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}

.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}

.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}

.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}

.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}

.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}

.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}

.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}

/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}

.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}

.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}

.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}

.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}

.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}

.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}

/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

.jp-icon-dot[fill] {
  fill: var(--jp-warn-color0);
}

.jp-jupyter-icon-color[fill] {
  fill: var(--jp-jupyter-icon-color, var(--jp-warn-color0));
}

.jp-notebook-icon-color[fill] {
  fill: var(--jp-notebook-icon-color, var(--jp-warn-color0));
}

.jp-json-icon-color[fill] {
  fill: var(--jp-json-icon-color, var(--jp-warn-color1));
}

.jp-console-icon-color[fill] {
  fill: var(--jp-console-icon-color, white);
}

.jp-console-icon-background-color[fill] {
  fill: var(--jp-console-icon-background-color, var(--jp-brand-color1));
}

.jp-terminal-icon-color[fill] {
  fill: var(--jp-terminal-icon-color, var(--jp-layout-color2));
}

.jp-terminal-icon-background-color[fill] {
  fill: var(
    --jp-terminal-icon-background-color,
    var(--jp-inverse-layout-color2)
  );
}

.jp-text-editor-icon-color[fill] {
  fill: var(--jp-text-editor-icon-color, var(--jp-inverse-layout-color3));
}

.jp-inspector-icon-color[fill] {
  fill: var(--jp-inspector-icon-color, var(--jp-inverse-layout-color3));
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* stylelint-disable selector-max-class, selector-max-compound-selectors */

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}

.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* stylelint-enable selector-max-class, selector-max-compound-selectors */

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) .jp-icon-hoverShow-content {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FormGroup-content fieldset {
  border: none;
  padding: 0;
  min-width: 0;
  width: 100%;
}

/* stylelint-disable selector-max-type */

.jp-FormGroup-content fieldset .jp-inputFieldWrapper > input,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper > select,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper > textarea {
  font-size: var(--jp-content-font-size2);
  border-color: var(--jp-input-border-color);
  border-style: solid;
  border-radius: var(--jp-border-radius);
  border-width: 1px;
  padding: 6px 8px;
  background: none;
  color: var(--jp-ui-font-color0);
  height: inherit;
}

.jp-FormGroup-content .jp-inputFieldWrapper > select > option {
  background-color: var(--jp-layout-color1);
}

.jp-FormGroup-content fieldset input[type='checkbox'] {
  position: relative;
  top: 2px;
  margin-left: 0;
}

.jp-FormGroup-content button.jp-mod-styled {
  cursor: pointer;
}

.jp-FormGroup-content .checkbox label {
  cursor: pointer;
  font-size: var(--jp-content-font-size1);
}

.jp-FormGroup-content .jp-root > fieldset > legend {
  display: none;
}

.jp-FormGroup-content .jp-root > fieldset > p {
  display: none;
}

/** copy of `input.jp-mod-styled:focus` style */
.jp-FormGroup-content fieldset input:focus,
.jp-FormGroup-content fieldset select:focus {
  -moz-outline-radius: unset;
  outline: var(--jp-border-width) solid var(--md-blue-500);
  outline-offset: -1px;
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FormGroup-content fieldset input:hover:not(:focus),
.jp-FormGroup-content fieldset select:hover:not(:focus) {
  background-color: var(--jp-border-color2);
}

/* stylelint-enable selector-max-type */

.jp-FormGroup-content .checkbox .field-description {
  /* Disable default description field for checkbox:
   because other widgets do not have description fields,
   we add descriptions to each widget on the field level.
  */
  display: none;
}

.jp-FormGroup-content #root__description {
  display: none;
}

.jp-FormGroup-content .jp-modifiedIndicator {
  width: 5px;
  background-color: var(--jp-brand-color2);
  margin-top: 0;
  margin-left: calc(var(--jp-private-settingeditor-modifier-indent) * -1);
  flex-shrink: 0;
}

.jp-FormGroup-content .jp-modifiedIndicator.jp-errorIndicator {
  background-color: var(--jp-error-color0);
  margin-right: 0.5em;
}

/* RJSF ARRAY style */

.jp-arrayFieldWrapper legend {
  font-size: var(--jp-content-font-size2);
  color: var(--jp-ui-font-color0);
  flex-basis: 100%;
  padding: 4px 0;
  font-weight: var(--jp-content-heading-font-weight);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-arrayFieldWrapper .field-description {
  padding: 4px 0;
  white-space: pre-wrap;
}

.jp-arrayFieldWrapper .array-item {
  width: 100%;
  border: 1px solid var(--jp-border-color2);
  border-radius: 4px;
  margin: 4px;
}

.jp-ArrayOperations {
  display: flex;
  margin-left: 8px;
}

.jp-ArrayOperationsButton {
  margin: 2px;
}

.jp-ArrayOperationsButton .jp-icon3[fill] {
  fill: var(--jp-ui-font-color0);
}

button.jp-ArrayOperationsButton.jp-mod-styled:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

/* RJSF form validation error */

.jp-FormGroup-content .validationErrors {
  color: var(--jp-error-color0);
}

/* Hide panel level error as duplicated the field level error */
.jp-FormGroup-content .panel.errors {
  display: none;
}

/* RJSF normal content (settings-editor) */

.jp-FormGroup-contentNormal {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-FormGroup-contentItem {
  margin-left: 7px;
  color: var(--jp-ui-font-color0);
}

.jp-FormGroup-contentNormal .jp-FormGroup-description {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-default {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-fieldLabel {
  font-size: var(--jp-content-font-size1);
  font-weight: normal;
  min-width: 120px;
}

.jp-FormGroup-contentNormal fieldset:not(:first-child) {
  margin-left: 7px;
}

.jp-FormGroup-contentNormal .field-array-of-string .array-item {
  /* Display `jp-ArrayOperations` buttons side-by-side with content except
    for small screens where flex-wrap will place them one below the other.
  */
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-objectFieldWrapper .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

/* RJSF compact content (metadata-form) */

.jp-FormGroup-content.jp-FormGroup-contentCompact {
  width: 100%;
}

.jp-FormGroup-contentCompact .form-group {
  display: flex;
  padding: 0.5em 0.2em 0.5em 0;
}

.jp-FormGroup-contentCompact
  .jp-FormGroup-compactTitle
  .jp-FormGroup-description {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color2);
}

.jp-FormGroup-contentCompact .jp-FormGroup-fieldLabel {
  padding-bottom: 0.3em;
}

.jp-FormGroup-contentCompact .jp-inputFieldWrapper .form-control {
  width: 100%;
  box-sizing: border-box;
}

.jp-FormGroup-contentCompact .jp-arrayFieldWrapper .jp-FormGroup-compactTitle {
  padding-bottom: 7px;
}

.jp-FormGroup-contentCompact
  .jp-objectFieldWrapper
  .jp-objectFieldWrapper
  .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

.jp-FormGroup-contentCompact ul.error-detail {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
  padding-inline-start: 1em;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-SidePanel {
  display: flex;
  flex-direction: column;
  min-width: var(--jp-sidebar-min-width);
  overflow-y: auto;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size1);
}

.jp-SidePanel-header {
  flex: 0 0 auto;
  display: flex;
  border-bottom: var(--jp-border-width) solid var(--jp-border-color2);
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin: 0;
  padding: 2px;
  text-transform: uppercase;
}

.jp-SidePanel-toolbar {
  flex: 0 0 auto;
}

.jp-SidePanel-content {
  flex: 1 1 auto;
}

.jp-SidePanel-toolbar,
.jp-AccordionPanel-toolbar {
  height: var(--jp-private-toolbar-height);
}

.jp-SidePanel-toolbar.jp-Toolbar-micro {
  display: none;
}

.lm-AccordionPanel .jp-AccordionPanel-title {
  box-sizing: border-box;
  line-height: 25px;
  margin: 0;
  display: flex;
  align-items: center;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  font-size: var(--jp-ui-font-size0);
}

.jp-AccordionPanel-title {
  cursor: pointer;
  user-select: none;
  -moz-user-select: none;
  -webkit-user-select: none;
  text-transform: uppercase;
}

.lm-AccordionPanel[data-orientation='horizontal'] > .jp-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleLabel {
  user-select: none;
  text-overflow: ellipsis;
  white-space: nowrap;
  overflow: hidden;
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleCollapser {
  transform: rotate(-90deg);
  margin: auto 0;
  height: 16px;
}

.jp-AccordionPanel-title.lm-mod-expanded .lm-AccordionPanel-titleCollapser {
  transform: rotate(0deg);
}

.lm-AccordionPanel .jp-AccordionPanel-toolbar {
  background: none;
  box-shadow: none;
  border: none;
  margin-left: auto;
}

.lm-AccordionPanel .lm-SplitPanel-handle:hover {
  background: var(--jp-layout-color3);
}

.jp-text-truncated {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation:
    load3 1s infinite linear,
    fadeIn 1s;
}

.jp-SpinnerContent::before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent::after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

input[type='checkbox'].jp-mod-styled:focus-visible {
  outline: var(--jp-border-width) solid var(--jp-input-active-border-color);
  outline-offset: 1px;
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper:not(.multiple) {
  height: 28px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

select.jp-mod-styled:not([multiple]) {
  height: 32px;
}

select.jp-mod-styled[multiple] {
  max-height: 200px;
  overflow-y: auto;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
  font-family: var(--jp-ui-font-family);
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-switch-color, var(--jp-border-color1));
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-switch-true-position-color, var(--jp-warn-color0));
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 8;
  overflow-x: hidden;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0;
  margin: 0;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0 6px;
  margin: 0;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent > span {
  padding: 0;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-WindowedPanel-outer {
  position: relative;
  overflow-y: auto;
}

.jp-WindowedPanel-inner {
  position: relative;
}

.jp-WindowedPanel-window {
  position: absolute;
  left: 0;
  right: 0;
  overflow: visible;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

body {
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
}

/* Disable native link decoration styles everywhere outside of dialog boxes */
a {
  text-decoration: unset;
  color: unset;
}

a:hover {
  text-decoration: unset;
  color: unset;
}

/* Accessibility for links inside dialog box text */
.jp-Dialog-content a {
  text-decoration: revert;
  color: var(--jp-content-link-color);
}

.jp-Dialog-content a:hover {
  text-decoration: revert;
}

/* Styles for ui-components */
.jp-Button {
  color: var(--jp-ui-font-color2);
  border-radius: var(--jp-border-radius);
  padding: 0 12px;
  font-size: var(--jp-ui-font-size1);

  /* Copy from blueprint 3 */
  display: inline-flex;
  flex-direction: row;
  border: none;
  cursor: pointer;
  align-items: center;
  justify-content: center;
  text-align: left;
  vertical-align: middle;
  min-height: 30px;
  min-width: 30px;
}

.jp-Button:disabled {
  cursor: not-allowed;
}

.jp-Button:empty {
  padding: 0 !important;
}

.jp-Button.jp-mod-small {
  min-height: 24px;
  min-width: 24px;
  font-size: 12px;
  padding: 0 7px;
}

/* Use our own theme for hover styles */
.jp-Button.jp-mod-minimal:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Button.jp-mod-minimal {
  background: none;
}

.jp-InputGroup {
  display: block;
  position: relative;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border: none;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
  padding-bottom: 0;
  padding-top: 0;
  padding-left: 10px;
  padding-right: 28px;
  position: relative;
  width: 100%;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  font-size: 14px;
  font-weight: 400;
  height: 30px;
  line-height: 30px;
  outline: none;
  vertical-align: middle;
}

.jp-InputGroup input:focus {
  box-shadow:
    inset 0 0 0 var(--jp-border-width) var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input:disabled {
  cursor: not-allowed;
  resize: block;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input:disabled ~ span {
  cursor: not-allowed;
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color2);
}

.jp-InputGroupAction {
  position: absolute;
  bottom: 1px;
  right: 0;
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
  cursor: not-allowed;
  resize: block;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled ~ span {
  cursor: not-allowed;
}

/* Use our own theme for hover and option styles */
/* stylelint-disable-next-line selector-max-type */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}

select {
  box-sizing: border-box;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-StatusBar-Widget {
  display: flex;
  align-items: center;
  background: var(--jp-layout-color2);
  min-height: var(--jp-statusbar-height);
  justify-content: space-between;
  padding: 0 10px;
}

.jp-StatusBar-Left {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-StatusBar-Middle {
  display: flex;
  align-items: center;
}

.jp-StatusBar-Right {
  display: flex;
  align-items: center;
  flex-direction: row-reverse;
}

.jp-StatusBar-Item {
  max-height: var(--jp-statusbar-height);
  margin: 0 2px;
  height: var(--jp-statusbar-height);
  white-space: nowrap;
  text-overflow: ellipsis;
  color: var(--jp-ui-font-color1);
  padding: 0 6px;
}

.jp-mod-highlighted:hover {
  background-color: var(--jp-layout-color3);
}

.jp-mod-clicked {
  background-color: var(--jp-brand-color1);
}

.jp-mod-clicked:hover {
  background-color: var(--jp-brand-color0);
}

.jp-mod-clicked .jp-StatusBar-TextItem {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-StatusBar-HoverItem {
  box-shadow: 0 4px 4px rgba(0, 0, 0, 0.25);
}

.jp-StatusBar-TextItem {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  line-height: 24px;
  color: var(--jp-ui-font-color1);
}

.jp-StatusBar-GroupItem {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-Statusbar-ProgressCircle svg {
  display: block;
  margin: 0 auto;
  width: 16px;
  height: 24px;
  align-self: normal;
}

.jp-Statusbar-ProgressCircle path {
  fill: var(--jp-inverse-layout-color3);
}

.jp-Statusbar-ProgressBar-progress-bar {
  height: 10px;
  width: 100px;
  border: solid 0.25px var(--jp-brand-color2);
  border-radius: 3px;
  overflow: hidden;
  align-self: center;
}

.jp-Statusbar-ProgressBar-progress-bar > div {
  background-color: var(--jp-brand-color2);
  background-image: linear-gradient(
    -45deg,
    rgba(255, 255, 255, 0.2) 25%,
    transparent 25%,
    transparent 50%,
    rgba(255, 255, 255, 0.2) 50%,
    rgba(255, 255, 255, 0.2) 75%,
    transparent 75%,
    transparent
  );
  background-size: 40px 40px;
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 14px;
  color: #fff;
  text-align: center;
  animation: jp-Statusbar-ExecutionTime-progress-bar 2s linear infinite;
}

.jp-Statusbar-ProgressBar-progress-bar p {
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  line-height: 10px;
  width: 100px;
}

@keyframes jp-Statusbar-ExecutionTime-progress-bar {
  0% {
    background-position: 0 0;
  }

  100% {
    background-position: 40px 40px;
  }
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  /* stylelint-disable-next-line csstree/validator */
  overflow: overlay;
  padding: 0 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow:
    inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color2);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty::after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0 8px;
  color: var(--jp-content-font-color3);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0;
  left: 0;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px 24px 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-content.jp-Dialog-content-small {
  max-width: 500px;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus {
  outline: 1px solid var(--jp-accept-color-normal, var(--jp-brand-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus {
  outline: 1px solid var(--jp-warn-color-normal, var(--jp-error-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline: 1px solid var(--jp-reject-color-normal, var(--md-grey-600));
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color1);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  align-items: center;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-checkbox {
  padding-right: 5px;
}

.jp-Dialog-spacer {
  flex: 1 1 auto;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error {
  padding: 6px;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error > pre {
  width: auto;
  padding: 10px;
  background: var(--jp-error-color3);
  border: var(--jp-border-width) solid var(--jp-error-color1);
  border-radius: var(--jp-border-radius);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  white-space: pre-wrap;
  word-wrap: break-word;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;
  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;
  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #a0f;
  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;
  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;
  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;
  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;
  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;
  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;
  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;
  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;
  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;
  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ff0;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;
  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;
  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;
  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;
  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;
  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;
  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-menu-panel-height: 27px;
}

.lm-Widget.lm-mod-hidden {
  display: none !important;
}

body {
  font-family: var(--jp-ui-font-family);
  background: var(--jp-layout-color3);
  margin: 0;
  padding: 0;
  overflow: hidden;
}

.jp-LabShell {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
}

.jp-LabShell.jp-mod-devMode {
  border-top: 4px solid red;
}

#jp-main-dock-panel {
  padding: 5px;
}

#jp-main-dock-panel[data-mode='single-document'] {
  padding: 0;
}

#jp-main-dock-panel[data-mode='single-document'] .jp-MainAreaWidget {
  border: none;
}

#jp-top-panel {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color0);
  background: var(--jp-layout-color1);
  display: flex;
  min-height: var(--jp-private-menubar-height);
  overflow: visible;

  /* relax lumino strict CSS contaiment to allow painting the menu bar item
  over the menu in order to create an illusion of partial border */
  contain: style size !important;
}

#jp-menu-panel {
  min-height: var(--jp-private-menu-panel-height);
  background: var(--jp-layout-color1);
}

#jp-down-stack {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-LabShell[data-shell-mode='single-document'] #jp-top-panel {
  border-bottom: none;
}

.jp-LabShell[data-shell-mode='single-document'] #jp-menu-panel {
  padding-left: calc(
    var(--jp-private-sidebar-tab-width) + var(--jp-border-width)
  );
  border-bottom: var(--jp-border-width) solid var(--jp-border-color0);

  /* Adjust min-height so open menus show up in the right place */
  min-height: calc(
    var(--jp-private-menu-panel-height) + var(--jp-border-width)
  );
}

#jp-bottom-panel {
  background: var(--jp-layout-color1);
  display: flex;
}

#jp-single-document-mode {
  margin: 0 8px;
  display: flex;
  align-items: center;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.lm-DataGrid {
  min-width: 64px;
  min-height: 64px;
  border: 1px solid #a0a0a0;
}

.lm-DataGrid-scrollCorner {
  background-color: #f0f0f0;
}

.lm-DataGrid-scrollCorner::after {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 1px;
  height: 1px;
  background-color: #a0a0a0;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| DockPanel
|----------------------------------------------------------------------------*/

.lm-DockPanel-widget,
.lm-TabPanel-stackedPanel {
  background: var(--jp-layout-color0);
  border-left: var(--jp-border-width) solid var(--jp-border-color1);
  border-right: var(--jp-border-width) solid var(--jp-border-color1);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
}

.lm-DockPanel-overlay {
  background: rgba(33, 150, 243, 0.1);
  border: var(--jp-border-width) dashed var(--jp-brand-color1);
  transition-property: top, left, right, bottom;
  transition-duration: 150ms;
  transition-timing-function: ease;
}

.lm-DockPanel-overlay.lm-mod-root-top,
.lm-DockPanel-overlay.lm-mod-root-left,
.lm-DockPanel-overlay.lm-mod-root-right,
.lm-DockPanel-overlay.lm-mod-root-bottom,
.lm-DockPanel-overlay.lm-mod-root-center {
  border-width: 2px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-menubar-height: 28px;
  --jp-private-menu-item-height: 24px;
}

/*-----------------------------------------------------------------------------
| MenuBar
|----------------------------------------------------------------------------*/

.lm-MenuBar {
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
}

.lm-MenuBar:hover {
  overflow-x: auto;
}

.lm-MenuBar-menu {
  top: calc(-2 * var(--jp-border-width));
}

.lm-MenuBar-item {
  padding: 0 8px;
  border-left: var(--jp-border-width) solid transparent;
  border-right: var(--jp-border-width) solid transparent;
  border-top: var(--jp-border-width) solid transparent;
  line-height: calc(var(--jp-private-menubar-height) - var(--jp-border-width));
}

.lm-MenuBar-content:focus-visible {
  outline-offset: -3px; /* this value is a compromise between Firefox, Chrome,
    and Safari over this outline's visibility and discretion */
}

.lm-MenuBar:focus-visible {
  outline: 1px solid var(--jp-accept-color-active, var(--jp-brand-color1));
  outline-offset: -1px;
}

.lm-MenuBar-menu:focus-visible,
.lm-MenuBar-item:focus-visible,
.lm-Menu-item:focus-visible {
  outline: unset;
  outline-offset: unset;
  -moz-outline-radius: unset;
}

.lm-MenuBar-item.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-MenuBar.lm-mod-active .lm-MenuBar-item.lm-mod-active {
  z-index: 10001;
  background: var(--jp-layout-color0);
  color: var(--jp-ui-font-color0);
  border-left: var(--jp-border-width) solid var(--jp-border-color1);
  border-right: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-elevation-z6);
}

/* stylelint-disable-next-line selector-max-class */
.jp-LabShell[data-shell-mode='single-document']
  .lm-MenuBar.lm-mod-active
  .lm-MenuBar-item.lm-mod-active {
  border-top: var(--jp-border-width) solid var(--jp-border-color1);
}

.lm-MenuBar-item.lm-mod-disabled {
  color: var(--jp-ui-font-color3);
}

.lm-MenuBar-item.lm-type-separator {
  margin: 2px;
  padding: 0;
  border: none;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
}

.lm-MenuBar-itemMnemonic {
  text-decoration: underline;
}

/*-----------------------------------------------------------------------------
| Menu
|----------------------------------------------------------------------------*/

.lm-Menu {
  z-index: 10000;
  padding: 4px 0;
  background: var(--jp-layout-color0);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
  font-size: var(--jp-ui-font-size1);
  box-shadow: var(--jp-elevation-z6);
}

.lm-Menu-item {
  min-height: var(--jp-private-menu-item-height);
  max-height: var(--jp-private-menu-item-height);
  padding: 0;
  line-height: var(--jp-private-menu-item-height);
}

.lm-Menu-item.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-Menu-item.lm-mod-disabled {
  color: var(--jp-ui-font-color3);
}

.lm-Menu-itemIcon {
  width: 21px;
  padding: 0 2px 0 4px;
  margin-top: -2px;
}

.lm-Menu-itemLabel {
  padding: 0 32px 0 2px;
}

.lm-Menu-itemMnemonic {
  text-decoration: underline;
}

.lm-Menu-itemShortcut {
  padding: 0;
}

.lm-Menu-itemSubmenuIcon {
  width: 18px;
  padding: 0 4px 0 0;
}

.lm-Menu-item[data-type='separator'] > div {
  padding: 0;
  height: 9px;
}

.lm-Menu-item[data-type='separator'] > div::after {
  content: '';
  display: block;
  position: relative;
  top: 4px;
  border-top: var(--jp-border-width) solid var(--jp-layout-color2);
}

/* gray out icon/caret for disabled menu items */
.lm-Menu-item.lm-mod-disabled > .lm-Menu-itemIcon,
.lm-Menu-item[data-type='submenu'].lm-mod-disabled > .lm-Menu-itemSubmenuIcon {
  opacity: 0.4;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0 solid transparent;
  border-right: 0 solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0 solid transparent;
  border-bottom: 0 solid transparent;
}

/*
 * Lumino
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  /* These need to be root because tabs get attached to the body during dragging. */
  --jp-private-horizontal-tab-height: 24px;
  --jp-private-horizontal-tab-width: 216px;
  --jp-private-horizontal-tab-active-top-border: 2px;
}

/*-----------------------------------------------------------------------------
| Tabs in the dock panel
|----------------------------------------------------------------------------*/

.lm-DockPanel-tabBar,
.lm-TabPanel-tabBar {
  overflow: visible;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
}

.lm-DockPanel-tabBar[data-orientation='horizontal'],
.lm-TabPanel-tabBar[data-orientation='horizontal'] {
  min-height: calc(
    var(--jp-private-horizontal-tab-height) + 2 * var(--jp-border-width)
  );
}

.lm-DockPanel-tabBar[data-orientation='vertical'] {
  min-width: 80px;
}

.lm-DockPanel-tabBar > .lm-TabBar-content,
.lm-TabPanel-tabBar > .lm-TabBar-content {
  align-items: flex-end;
  min-width: 0;
  min-height: 0;
}

.lm-DockPanel-tabBar .lm-TabBar-tab,
.lm-TabPanel-tabBar .lm-TabBar-tab {
  flex: 0 1 var(--jp-private-horizontal-tab-width);
  align-items: center;
  min-height: calc(
    var(--jp-private-horizontal-tab-height) + 2 * var(--jp-border-width)
  );
  min-width: 0;
  margin-left: calc(-1 * var(--jp-border-width));
  line-height: var(--jp-private-horizontal-tab-height);
  padding: 0 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-border-color1);
  border-bottom: none;
  position: relative;
}

.lm-DockPanel-tabBar .lm-TabBar-tab:hover:not(.lm-mod-current),
.lm-TabPanel-tabBar .lm-TabBar-tab:hover:not(.lm-mod-current) {
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
}

.lm-DockPanel-tabBar .lm-TabBar-tab:not(.lm-mod-current)::after,
.lm-DockPanel-tabBar .lm-TabBar-addButton::after {
  position: absolute;
  content: '';
  bottom: 0;
  left: calc(-1 * var(--jp-border-width));
  width: calc(100% + 2 * var(--jp-border-width));
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
}

.lm-DockPanel-tabBar .lm-TabBar-tab:first-child,
.lm-TabPanel-tabBar .lm-TabBar-tab:first-child {
  margin-left: 0;
}

/* This is a current tab of a tab bar in the dock panel: each tab bar has 1. */
.lm-DockPanel-tabBar .lm-TabBar-tab.lm-mod-current {
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
}

.lm-TabPanel-tabBar .lm-TabBar-tab.lm-mod-current {
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
}

/* This is the main application level current tab: only 1 exists. */
.lm-DockPanel-tabBar .lm-TabBar-tab.jp-mod-current::before {
  position: absolute;
  top: calc(-1 * var(--jp-border-width) + 1px);
  left: calc(-1 * var(--jp-border-width));
  content: '';
  height: var(--jp-private-horizontal-tab-active-top-border);
  width: calc(100% + 2 * var(--jp-border-width));
  background: var(--jp-brand-color1);
}

/* This is the left tab bar current tab: only 1 exists. */
.lm-TabBar-tab.lm-mod-current {
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
}

.lm-DockPanel-tabBar .lm-TabBar.lm-mod-left .lm-TabBar-tab,
.lm-DockPanel-tabBar .lm-TabBar.lm-mod-right .lm-TabBar-tab {
  flex: 0 1 40px;
  margin-top: -1px;
  line-height: 40px;
}

.lm-DockPanel-tabBar .lm-TabBar.lm-mod-left .lm-TabBar-tab {
  border-right: none;
}

.lm-DockPanel-tabBar .lm-TabBar.lm-mod-right .lm-TabBar-tab {
  border-left: none;
}

.lm-DockPanel-tabBar .lm-TabBar.lm-mod-left .lm-TabBar-tab:first-child,
.lm-DockPanel-tabBar .lm-TabBar.lm-mod-right .lm-TabBar-tab:first-child {
  margin-top: 0;
}

/* stylelint-disable selector-max-class */

.lm-DockPanel-tabBar .lm-TabBar.lm-mod-left .lm-TabBar-tab.lm-mod-current,
.lm-DockPanel-tabBar .lm-TabBar.lm-mod-right .lm-TabBar-tab.lm-mod-current {
  min-width: 80px;
  max-width: 80px;
}

.lm-DockPanel-tabBar .lm-TabBar.lm-mod-right .lm-TabBar-tab.lm-mod-current {
  transform: translateX(-1px);
}

.lm-DockPanel-tabBar .lm-TabBar-tab .lm-TabBar-tabIcon,
.lm-TabBar-tab.lm-mod-drag-image .lm-TabBar-tabIcon,
.lm-TabPanel-tabBar .lm-TabBar-tab .lm-TabBar-tabIcon {
  width: 14px;
  background-position: left center;
  background-repeat: no-repeat;
  background-size: 14px;
  margin-right: 4px;
}

/* stylelint-enable selector-max-class */

.lm-TabBar-tab.lm-mod-drag-image {
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  border: var(--jp-border-width) solid var(--jp-border-color1);
  border-top: var(--jp-border-width) solid var(--jp-brand-color1);
  box-shadow: var(--jp-elevation-z4);
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-horizontal-tab-height);
  min-height: var(--jp-private-horizontal-tab-height);
  min-width: var(--jp-private-horizontal-tab-width);
  padding: 0 10px;
  transform: translateX(-40%) translateY(-58%);
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-flat-button-height: 24px;
  --jp-flat-button-padding: 8px 12px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button {
  border-radius: var(--jp-border-radius);
}

button:focus-visible {
  outline: 1px solid var(--jp-accept-color-active, var(--jp-brand-color1));
  outline-offset: -1px;
}

button.jp-mod-styled.jp-mod-accept {
  background: var(--jp-accept-color-normal, var(--md-blue-500));
  border: 0;
  color: white;
}

button.jp-mod-styled.jp-mod-accept:hover {
  background: var(--jp-accept-color-hover, var(--md-blue-600));
}

button.jp-mod-styled.jp-mod-accept:active {
  background: var(--jp-accept-color-active, var(--md-blue-700));
}

button.jp-mod-styled.jp-mod-accept:focus-visible {
  outline: 1px solid var(--jp-accept-color-active, var(--jp-brand-color1));
}

button.jp-mod-styled.jp-mod-reject {
  background: var(--jp-reject-color-normal, var(--md-grey-500));
  border: 0;
  color: white;
}

button.jp-mod-styled.jp-mod-reject:hover {
  background: var(--jp-reject-color-hover, var(--md-grey-600));
}

button.jp-mod-styled.jp-mod-reject:active {
  background: var(--jp-reject-color-active, var(--md-grey-700));
}

button.jp-mod-styled.jp-mod-reject:focus-visible {
  outline: 1px solid var(--jp-reject-color-active, var(--md-grey-700));
}

button.jp-mod-styled.jp-mod-warn {
  background: var(--jp-warn-color-normal, var(--jp-error-color1));
  border: 0;
  color: white;
}

button.jp-mod-styled.jp-mod-warn:hover {
  background: var(--jp-warn-color-hover, var(--md-red-600));
}

button.jp-mod-styled.jp-mod-warn:active {
  background: var(--jp-warn-color-active, var(--md-red-700));
}

button.jp-mod-styled.jp-mod-warn:focus-visible {
  outline: 1px solid var(--jp-warn-color-active, var(--md-red-700));
}

.jp-Button-flat {
  text-decoration: none;
  padding: var(--jp-flat-button-padding);
  font-weight: 500;
  background-color: transparent;
  height: var(--jp-private-running-shutdown-button-height);
  line-height: var(--jp-private-running-shutdown-button-height);
  transition: background-color 0.1s ease;
  border-radius: 2px;
}

.jp-Button-flat:focus {
  border: none;
  box-shadow: none;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-sidebar-tab-width: 32px;
}

/*-----------------------------------------------------------------------------
| SideBar
|----------------------------------------------------------------------------*/

.jp-SideBar {
  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

.jp-SideBar.lm-TabBar,
#jp-down-stack .lm-TabBar {
  color: var(--jp-ui-font-color2);
  background: var(--jp-layout-color2);
  font-size: var(--jp-ui-font-size1);
  overflow: visible;
}

.jp-SideBar.lm-TabBar {
  min-width: calc(var(--jp-private-sidebar-tab-width) + var(--jp-border-width));
  max-width: calc(var(--jp-private-sidebar-tab-width) + var(--jp-border-width));
  display: block;
}

.jp-SideBar .lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  align-items: stretch;
  list-style-type: none;
  height: var(--jp-private-sidebar-tab-width);
}

.jp-SideBar .lm-TabBar-tab {
  padding: 16px 0;
  border: none;
  overflow: visible;
  flex-direction: column;
  position: relative;
}

.jp-SideBar .lm-TabBar-tab.lm-mod-current::after {
  /* Internal border override pseudo-element */
  position: absolute;
  content: '';
  bottom: 0;
  right: 0;
  top: 0;
  left: 0;
  border: var(--jp-border-width) solid var(--jp-layout-color1);
}

.jp-SideBar .lm-TabBar-tab:not(.lm-mod-current),
#jp-down-stack .lm-TabBar-tab:not(.lm-mod-current) {
  background: var(--jp-layout-color2);
}

.jp-SideBar .lm-TabBar-tabIcon.jp-SideBar-tabIcon {
  min-width: 20px;
  min-height: 20px;
  background-size: 20px;
  display: inline-block;
  vertical-align: middle;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-SideBar .lm-TabBar-tabLabel {
  line-height: var(--jp-private-sidebar-tab-width);
}

.jp-SideBar .lm-TabBar-tab:hover:not(.lm-mod-current),
#jp-down-stack .lm-TabBar-tab:hover:not(.lm-mod-current) {
  background: var(--jp-layout-color1);
}

.jp-SideBar.lm-TabBar::after {
  /* Internal border pseudo-element */
  position: absolute;
  content: '';
  bottom: 0;
  right: 0;
  top: 0;
  left: 0;
  pointer-events: none;
}

/* Borders */

/* stylelint-disable selector-max-class */

.jp-SideBar.lm-TabBar .lm-TabBar-tab + .lm-TabBar-tab {
  border-top: var(--jp-border-width) solid var(--jp-layout-color2);
}

.jp-SideBar.lm-TabBar .lm-TabBar-tab.lm-mod-current + .lm-TabBar-tab {
  border-top: var(--jp-border-width) solid var(--jp-border-color0);
}

.jp-SideBar.lm-TabBar .lm-TabBar-tab + .lm-TabBar-tab.lm-mod-current {
  border-top: var(--jp-border-width) solid var(--jp-border-color0);
}

.jp-SideBar.lm-TabBar .lm-TabBar-tab.lm-mod-current:last-child {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color0);
}

.jp-SideBar.lm-TabBar .lm-TabBar-tabLabel {
  writing-mode: vertical-rl;
}

/* Left */

/* Borders */

.jp-SideBar.lm-TabBar.jp-mod-left .lm-TabBar-content {
  /* Internal border spacing */
  margin-right: var(--jp-border-width);
}

.jp-SideBar.lm-TabBar.jp-mod-left .lm-TabBar-tab.lm-mod-current::after {
  /* Internal border override */
  right: calc(-1 * var(--jp-border-width));
}

.jp-SideBar.lm-TabBar.jp-mod-left::after {
  /* Internal border */
  border-right: var(--jp-border-width) solid var(--jp-border-color0);
}

/* Transforms */

.jp-SideBar.lm-TabBar.jp-mod-left .lm-TabBar-tabLabel {
  transform: rotate(180deg);
}

/* Right */

/* Borders */

.jp-SideBar.lm-TabBar.jp-mod-right .lm-TabBar-content {
  /* Internal border spacing */
  margin-left: var(--jp-border-width);
}

.jp-SideBar.lm-TabBar.jp-mod-right .lm-TabBar-tab.lm-mod-current::after {
  /* Internal border override */
  left: calc(-1 * var(--jp-border-width));
}

.jp-SideBar.lm-TabBar.jp-mod-right::after {
  /* Internal border */
  border-left: var(--jp-border-width) solid var(--jp-border-color0);
}

/* Down */

/* Borders */

#jp-down-stack > .lm-TabBar {
  border-top: var(--jp-border-width) solid var(--jp-border-color0);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color0);
}

#jp-down-stack > .lm-TabBar .lm-TabBar-tab {
  border-left: none;
  border-right: none;
}

#jp-down-stack > .lm-TabBar .lm-TabBar-tab.lm-mod-current {
  border: var(--jp-border-width) solid var(--jp-border-color1);
  border-bottom: none;
  transform: translateY(var(--jp-border-width));
}

#jp-down-stack > .lm-TabBar .lm-TabBar-tab.lm-mod-current:first-child {
  border: none;
  border-right: var(--jp-border-width) solid var(--jp-border-color1);
}

/* stylelint-enable selector-max-class */

/* Stack panels */

#jp-left-stack > .lm-Widget,
#jp-right-stack > .lm-Widget {
  min-width: var(--jp-sidebar-min-width);
  background-color: var(--jp-layout-color1);
}

#jp-right-stack {
  border-left: var(--jp-border-width) solid var(--jp-border-color1);
}

#jp-left-stack {
  border-right: var(--jp-border-width) solid var(--jp-border-color1);
}

#jp-down-stack > .lm-TabPanel-stackedPanel {
  border: none;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-title-panel-height: 28px;
}

#jp-title-panel {
  min-height: var(--jp-private-title-panel-height);
  width: 100%;
  display: flex;
  background: var(--jp-layout-color1);
}

#jp-title-panel-title {
  flex: 1 1 auto;
  margin-left: 8px;
}

#jp-title-panel-title input {
  background: transparent;
  margin: 0;
  height: 28px;
  width: 100%;
  box-sizing: border-box;
  border: none;
  font-size: 18px;
  font-weight: normal;
  font-family: var(--jp-ui-font-family);
  line-height: var(--jp-private-title-panel-height);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-skiplink-wrapper {
  overflow: visible;
}

.jp-skiplink {
  position: absolute;
  top: -100em;
}

.jp-skiplink:focus-within {
  position: absolute;
  z-index: 10000;
  top: 0;
  left: 46%;
  margin: 0 auto;
  padding: 1em;
  width: 15%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
  text-align: center;
}

.jp-skiplink:focus-within a {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-topbar-height: 28px;
  /* Override the layout-2 color for the dark theme */
  --md-grey-800: #323232;
  --jp-notebook-max-width: 1200px;
}

body {
  margin: 0;
  padding: 0;
  background: var(--jp-layout-color2);
}

#main {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
}

#top-panel-wrapper {
  min-height: calc(1.5 * var(--jp-private-topbar-height));
  border-bottom: var(--jp-border-width) solid var(--jp-border-color0);
  background: var(--jp-layout-color1);
}

#top-panel {
  display: flex;
  min-height: calc(1.5 * var(--jp-private-topbar-height));
  padding-left: 5px;
  padding-right: 5px;
  margin-left: auto;
  margin-right: auto;
  max-width: 1200px;
}

#menu-panel-wrapper {
  min-height: var(--jp-private-topbar-height);
  background: var(--jp-layout-color1);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color0);
  box-shadow: var(--jp-elevation-z1);
}

#menu-panel {
  display: flex;
  min-height: var(--jp-private-topbar-height);
  background: var(--jp-layout-color1);
  padding-left: 5px;
  padding-right: 5px;
  margin-left: auto;
  margin-right: auto;
  max-width: var(--jp-notebook-max-width);
}

#main-panel {
  margin-left: auto;
  margin-right: auto;
  max-width: var(--jp-notebook-max-width);
}

#spacer-widget-top {
  min-height: 16px;
}

/* Only edit pages should have a bottom space */

body[data-notebook='edit'] #spacer-widget-bottom {
  min-height: 16px;
}

/* Special case notebooks as document oriented pages */

[data-notebook]:not(body[data-notebook='notebooks']) #main-panel {
  box-shadow: var(--jp-elevation-z4);
}

.jp-TreePanel > .lm-TabPanel-stackedPanel {
  box-shadow: var(--jp-elevation-z4);
}

body[data-notebook='notebooks'] #main-panel {
  margin-left: unset;
  margin-right: unset;
  max-width: unset;
}

body[data-notebook='notebooks'] #spacer-widget-top {
  min-height: unset;
}

#main-panel > .jp-TreePanel {
  padding: 0px 5px;
}

@media only screen and (max-width: 760px) {
  #main-panel > .jp-TreePanel {
    margin: 0px -5px;
  }
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|
| Adapted from JupyterLab's packages/application/style/sidepanel.css.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-sidebar-tab-width: 32px;
}

/*-----------------------------------------------------------------------------
| SideBar
|----------------------------------------------------------------------------*/

/* Stack panels */

#jp-right-stack,
#jp-left-stack {
  display: flex;
  flex-direction: column;
  min-width: var(--jp-sidebar-min-width);
}

#jp-left-stack .jp-SidePanel-collapse,
#jp-right-stack .jp-SidePanel-collapse {
  display: flex;
  flex: 0 0 auto;
  min-height: 0;
  padding: 0;
}

#jp-left-stack .jp-SidePanel-collapse {
  justify-content: right;
}

#jp-right-stack .jp-SidePanel-collapse {
  justify-content: left;
}

#jp-left-stack .lm-StackedPanel,
#jp-right-stack .lm-StackedPanel {
  flex: 1 1 auto;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-NotebookSpacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-MainAreaWidget {
  height: 100%;
}

.jp-Toolbar > .jp-Toolbar-item {
  height: unset;
}

#jp-UserMenu {
  flex: 0 0 auto;
  display: flex;
  text-align: center;
  margin-top: 8px;
}

.jp-MimeDocument .jp-RenderedJSON {
  background: var(--jp-layout-color0);
}

/* Hide the stub toolbar that appears above terminals and documents */

.jp-MainAreaWidget > .jp-Toolbar-micro {
  display: none;
}

#jp-NotebookLogo {
  /* bring logo to the front so it is selectable by tab*/
  z-index: 10;
}
</style><style></style><style>.jp-Document {
  height: 100%;
}
</style><style></style><style>.jp-AboutNotebook .jp-Dialog-header {
  justify-content: center;
  padding: 0;
}

.jp-AboutNotebook-header {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: var(--jp-flat-button-padding);
}

.jp-AboutNotebook-header-text {
  margin-left: 16px;
}

.jp-AboutNotebook-version {
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  padding-bottom: 30px;
  font-weight: 400;
  letter-spacing: 0.4px;
  line-height: 1.12;
  min-width: 360px;
  text-align: center;
}

.jp-AboutNotebook-body {
  display: flex;
  font-size: var(--jp-ui-font-size2);
  padding: var(--jp-flat-button-padding);
  color: var(--jp-ui-font-color1);
  text-align: center;
  flex-direction: column;
  min-width: 360px;
  overflow: hidden;
}

.jp-AboutNotebook-about-body pre {
  white-space: pre-wrap;
}

.jp-AboutNotebook-about-externalLinks {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  align-items: flex-start;
  color: var(--jp-warn-color0);
}

.jp-AboutNotebook-about-copyright {
  padding-top: 25px;
}
</style><style>:root {
  --jp-notebook-toolbar-margin-bottom: 20px;
  --jp-notebook-padding-offset: 20px;

  --jp-kernel-status-padding: 5px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
  Document oriented look for the notebook.
  This includes changes to the look and feel of the JupyterLab Notebook
  component like:
  - scrollbar to the right of the page
  - drop shadow on the notebook
  - smaller empty space at the bottom of the notebook
  - compact view on mobile
*/

/* Keep the notebook centered on the page */

body[data-notebook='notebooks'] .jp-NotebookPanel-toolbar {
  padding-left: calc(calc(100% - var(--jp-notebook-max-width)) * 0.5);
  padding-right: calc(calc(100% - var(--jp-notebook-max-width)) * 0.5);
}

body[data-notebook='notebooks'] .jp-Notebook {
  padding-top: unset;
  padding-left: calc(calc(100% - var(--jp-notebook-max-width)) * 0.5);
  padding-right: calc(
    calc(
        100% - var(--jp-notebook-max-width) - var(--jp-notebook-padding-offset)
      ) * 0.5
  );
  background: var(--jp-layout-color2);
}

body[data-notebook='notebooks'] .jp-WindowedPanel-inner {
  margin-top: var(--jp-notebook-toolbar-margin-bottom);
}

body[data-notebook='notebooks'] .jp-Notebook-cell {
  background: var(--jp-layout-color0);
  padding-left: calc(2 * var(--jp-cell-padding));
  padding-right: calc(2 * var(--jp-cell-padding));
}

/* Empty space at the bottom of the notebook (similar to classic) */
body[data-notebook='notebooks'] .jp-Notebook.jp-mod-scrollPastEnd::after {
  min-height: 100px;
}

/* Fix background colors */

body[data-notebook='notebooks'] .jp-Notebook > * {
  background: var(--jp-layout-color0);
}

body[data-notebook='notebooks']
  .jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: var(--jp-layout-color0) !important;
}

/**
  Extra padding to the first and and last cell of the notebook.
  TODO: revisit when https://github.com/jupyterlab/jupyterlab/issues/13151 is fixed
*/
.jp-Notebook-cell[data-windowed-list-index='0'] {
  padding-top: calc(2 * var(--jp-notebook-padding));
}

body[data-notebook='notebooks'] .jp-WindowedPanel-window > *:last-child {
  padding-bottom: calc(2 * var(--jp-notebook-padding));
}

body[data-notebook='notebooks']
  .jp-Notebook
  .jp-Notebook-cell:not(:first-child)::before {
  content: ' ';
  height: 100%;
  position: absolute;
  top: 0;
  width: 11px;
}

/* Cell toolbar adjustments */

body[data-notebook='notebooks'] .jp-cell-toolbar {
  background: unset;
  box-shadow: unset;
}

body[data-notebook='notebooks']
  .jp-RawCell[data-windowed-list-index='0']
  .jp-cell-toolbar,
body[data-notebook='notebooks']
  .jp-MarkdownCell[data-windowed-list-index='0']
  .jp-cell-toolbar {
  top: calc(2 * var(--jp-notebook-padding));
}

/** first code cell on mobile
    (keep the selector above the media query)
*/
body[data-notebook='notebooks']
  .jp-CodeCell[data-windowed-list-index='0']
  .jp-cell-toolbar {
  top: unset;
}

@media only screen and (max-width: 760px) {
  /* first code cell on mobile */
  body[data-notebook='notebooks']
    .jp-CodeCell[data-windowed-list-index='0']
    .jp-cell-toolbar {
    top: var(--jp-notebook-padding);
  }

  body[data-notebook='notebooks'] .jp-MarkdownCell .jp-cell-toolbar,
  body[data-notebook='notebooks'] .jp-RawCell .jp-cell-toolbar {
    top: calc(0.5 * var(--jp-notebook-padding));
  }
}

/* Tweak the notebook footer (to add a new cell) */
body[data-notebook='notebooks'] .jp-Notebook-footer {
  width: 100%;
  margin-left: unset;
  background: unset;
}

/* Mobile View */

body[data-format='mobile'] .jp-NotebookCheckpoint {
  display: none;
}

body[data-format='mobile'] .jp-Notebook > *:first-child {
  margin-top: 0;
}

body[data-format='mobile'] .jp-ToolbarButton .jp-DebuggerBugButton {
  display: none;
}

/* Virtual Notebook fixes */

body[data-notebook='notebooks'] .jp-WindowedPanel-window {
  background: var(--jp-layout-color0);
}

/* Notebook box shadow */

body[data-notebook='notebooks'] .jp-Notebook > *:first-child:not(:last-child) {
  box-shadow: var(--jp-elevation-z4);
}

body[data-notebook='notebooks']
  .jp-Notebook
  > *:first-child:last-child::before {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  right: 0;
  box-shadow: 0px 0px 12px 1px var(--jp-shadow-umbra-color);
}

body[data-notebook='notebooks']
  .jp-Notebook
  .jp-Notebook-cell:not(:first-child)::after,
body[data-notebook='notebooks']
  .jp-Notebook
  .jp-Notebook-cell:not(:first-child)::before {
  content: ' ';
  height: 100%;
  position: absolute;
  top: 0;
  width: 11px;
}

/* Additional customizations of the components on the notebook page */

.jp-NotebookKernelLogo {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  text-align: center;
  margin-right: 8px;
}

.jp-NotebookKernelLogo img {
  max-width: 28px;
  max-height: 28px;
  display: flex;
}

.jp-NotebookKernelStatus {
  margin: 0;
  font-weight: normal;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  font-family: var(--jp-ui-font-family);
  line-height: var(--jp-private-title-panel-height);
  padding-left: var(--jp-kernel-status-padding);
  padding-right: var(--jp-kernel-status-padding);
}

.jp-NotebookKernelStatus-error {
  background-color: var(--jp-error-color0);
}

.jp-NotebookKernelStatus-warn {
  background-color: var(--jp-warn-color0);
}

.jp-NotebookKernelStatus-info {
  background-color: var(--jp-info-color0);
}

.jp-NotebookKernelStatus-fade {
  animation: 0.5s fade-out forwards;
}

.jp-NotebookTrustedStatus {
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  margin-top: 4px;
  margin-bottom: 4px;
  border: solid 1px var(--jp-border-color2);
  cursor: help;
}

.jp-NotebookTrustedStatus-not-trusted {
  cursor: pointer;
}

@keyframes fade-out {
  0% {
    opacity: 1;
  }
  100% {
    opacity: 0;
  }
}

#jp-title h1 {
  cursor: pointer;
  font-size: 18px;
  margin: 0;
  font-weight: normal;
  color: var(--jp-ui-font-color0);
  font-family: var(--jp-ui-font-family);
  line-height: calc(1.5 * var(--jp-private-title-panel-height));
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

#jp-title h1:hover {
  background: var(--jp-layout-color2);
}

.jp-NotebookCheckpoint {
  font-size: 14px;
  margin-left: 5px;
  margin-right: 5px;
  font-weight: normal;
  color: var(--jp-ui-font-color0);
  font-family: var(--jp-ui-font-family);
  line-height: calc(1.5 * var(--jp-private-title-panel-height));
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

.jp-skiplink {
  position: absolute;
  top: -100em;
}

.jp-skiplink:focus-within {
  position: absolute;
  z-index: 10000;
  top: 0;
  left: 46%;
  margin: 0 auto;
  padding: 1em;
  width: 15%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
  text-align: center;
}

.jp-skiplink:focus-within a {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}
</style><style></style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-cursor-backdrop {
  top: 0px;
  left: 0px;
  position: fixed;
  width: 200px;
  height: 200px;
  margin-top: -100px;
  margin-left: -100px;
  will-change: transform;
  z-index: 100;
  scrollbar-width: none;
  -ms-overflow-style: none;
  overflow: scroll;
}

.lm-cursor-backdrop::after {
  content: '';
  height: 1200px;
  width: 1200px;
  display: block;
}

.lm-cursor-backdrop::-webkit-scrollbar {
  display: none;
}

.lm-mod-drag-image {
  top: 0px;
  left: 0px;
  will-change: transform;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser .jp-SidePanel-content {
  display: flex;
  flex-direction: column;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  flex-wrap: wrap;
  row-gap: 12px;
  border-bottom: none;
  height: auto;
  margin: 8px 12px 0;
  box-shadow: none;
  padding: 0;
  justify-content: flex-start;
}

.jp-FileBrowser-Panel {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0 2px;
  padding: 0 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0;
  padding-right: 2px;
  align-items: center;
  height: unset;
}

.jp-FileBrowser-toolbar > .jp-Toolbar-item .jp-ToolbarButtonComponent {
  width: 40px;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: var(--jp-error-color1);
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileSize-hidden {
  display: none;
}

.jp-FileBrowser .lm-AccordionPanel > h3:first-child {
  display: none;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  align-items: center;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-DirListing-headerItem.jp-id-filesize {
  flex: 0 0 75px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-checkboxWrapper {
  /* Increases hit area of checkbox. */
  padding: 4px;
}

.jp-DirListing-header
  .jp-DirListing-checkboxWrapper
  + .jp-DirListing-headerItem {
  padding-left: 4px;
}

.jp-DirListing-content .jp-DirListing-checkboxWrapper {
  position: relative;
  left: -4px;
  margin: -4px 0 -4px -8px;
}

.jp-DirListing-checkboxWrapper.jp-mod-visible {
  visibility: visible;
}

/* For devices that support hovering, hide checkboxes until hovered, selected...
*/
@media (hover: hover) {
  .jp-DirListing-checkboxWrapper {
    visibility: hidden;
  }

  .jp-DirListing-item:hover .jp-DirListing-checkboxWrapper,
  .jp-DirListing-item.jp-mod-selected .jp-DirListing-checkboxWrapper {
    visibility: visible;
  }
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemText:focus {
  outline-width: 2px;
  outline-color: var(--jp-inverse-layout-color1);
  outline-style: solid;
  outline-offset: 1px;
}

.jp-DirListing-item.jp-mod-selected .jp-DirListing-itemText:focus {
  outline-color: var(--jp-layout-color1);
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-itemFileSize {
  flex: 0 0 90px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon::before {
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon::before {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}
</style><style>.jp-FileBrowser {
  height: 100%;
}

.lm-TabPanel {
  height: 100%;
}

.jp-TreePanel .lm-TabPanel-tabBar {
  overflow: visible;
  min-height: 32px;
  border-bottom: unset;
  height: var(--jp-private-toolbar-height);
}

.jp-TreePanel .lm-TabBar-content {
  height: 100%;
}

.jp-TreePanel .lm-TabBar-tab {
  flex: 0 1 auto;
  color: var(--jp-ui-font-color0);
  font-size: var(--jp-ui-font-size1);
  height: 100%;
}

.jp-TreePanel .lm-TabBar-tabLabel {
  padding-left: 5px;
  padding-right: 5px;
}

.jp-FileBrowser-toolbar.jp-Toolbar .jp-ToolbarButtonComponent {
  width: unset;
}

.jp-FileBrowser-toolbar > .jp-Toolbar-item {
  flex-direction: column;
  justify-content: center;
}

.jp-DropdownMenu .lm-MenuBar-itemIcon svg {
  vertical-align: sub;
}

button[data-command='filebrowser:refresh'] .jp-ToolbarButtonComponent-label {
  display: none;
}

.jp-TreePanel .lm-TabBar-tabIcon svg {
  vertical-align: sub;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar .jp-Toolbar-item.jp-DropdownMenu,
.jp-FileBrowser-toolbar .jp-Toolbar-item.jp-ToolbarButton,
.jp-FileBrowser-toolbar .jp-Toolbar-item.jp-CommandToolbarButton {
  border: solid 1px var(--jp-border-color2);
  margin: 1px;
  padding: 0px;
}

.jp-FileBrowser-toolbar button.jp-ToolbarButtonComponent {
  height: var(--jp-flat-button-height);
}

.jp-FileBrowser-filterBox {
  padding: 0;
  flex: 0 0 auto;
}

.jp-FileBrowser-filterBox input {
  line-height: 24px;
}

.jp-DirListing-content .jp-DirListing-checkboxWrapper {
  visibility: visible;
}

/* Action buttons */

.jp-FileBrowser-toolbar > .jp-FileActions.jp-Toolbar-item {
  display: flex;
  flex-direction: row;
}

.jp-FileActions .jp-ToolbarButtonComponent-icon {
  display: none;
}

.jp-FileActions .jp-ToolbarButtonComponent[data-command='filebrowser:delete'] {
  background-color: var(--jp-error-color1);
}

.jp-FileActions
  .jp-ToolbarButtonComponent[data-command='filebrowser:delete']
  .jp-ToolbarButtonComponent-label {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-FileBrowser-toolbar .jp-FileActions .jp-ToolbarButtonComponent {
  border: solid 1px var(--jp-border-color2);
  margin: 1px;
  min-height: 100%;
}
</style><style></style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2019, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-PropertyInspector {
  display: flex;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
}

.jp-PropertyInspector-content {
  flex-grow: 1;
}

.jp-PropertyInspector-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
}

.jp-PropertyInspector-placeholderContent {
  padding: 8px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

#jp-MainLogo {
  width: calc(var(--jp-private-sidebar-tab-width) + var(--jp-border-width));
}

#jp-top-bar {
  --jp-private-toolbar-height: var(--jp-private-menu-panel-height);

  flex: 1 1 auto;
  padding: 0 2px;
  box-shadow: none;
  border: none;
  align-items: center;
}
</style><style>:root{--toastify-color-light:#fff;--toastify-color-dark:#121212;--toastify-color-info:#3498db;--toastify-color-success:#07bc0c;--toastify-color-warning:#f1c40f;--toastify-color-error:#e74c3c;--toastify-color-transparent:hsla(0,0%,100%,.7);--toastify-icon-color-info:var(--toastify-color-info);--toastify-icon-color-success:var(--toastify-color-success);--toastify-icon-color-warning:var(--toastify-color-warning);--toastify-icon-color-error:var(--toastify-color-error);--toastify-toast-width:320px;--toastify-toast-background:#fff;--toastify-toast-min-height:64px;--toastify-toast-max-height:800px;--toastify-font-family:sans-serif;--toastify-z-index:9999;--toastify-text-color-light:#757575;--toastify-text-color-dark:#fff;--toastify-text-color-info:#fff;--toastify-text-color-success:#fff;--toastify-text-color-warning:#fff;--toastify-text-color-error:#fff;--toastify-spinner-color:#616161;--toastify-spinner-color-empty-area:#e0e0e0;--toastify-color-progress-light:linear-gradient(90deg,#4cd964,#5ac8fa,#007aff,#34aadc,#5856d6,#ff2d55);--toastify-color-progress-dark:#bb86fc;--toastify-color-progress-info:var(--toastify-color-info);--toastify-color-progress-success:var(--toastify-color-success);--toastify-color-progress-warning:var(--toastify-color-warning);--toastify-color-progress-error:var(--toastify-color-error)}.Toastify__toast-container{z-index:var(--toastify-z-index);-webkit-transform:translateZ(var(--toastify-z-index));position:fixed;padding:4px;width:var(--toastify-toast-width);box-sizing:border-box;color:#fff}.Toastify__toast-container--top-left{top:1em;left:1em}.Toastify__toast-container--top-center{top:1em;left:50%;transform:translateX(-50%)}.Toastify__toast-container--top-right{top:1em;right:1em}.Toastify__toast-container--bottom-left{bottom:1em;left:1em}.Toastify__toast-container--bottom-center{bottom:1em;left:50%;transform:translateX(-50%)}.Toastify__toast-container--bottom-right{bottom:1em;right:1em}@media only screen and (max-width:480px){.Toastify__toast-container{width:100vw;padding:0;left:0;margin:0}.Toastify__toast-container--top-center,.Toastify__toast-container--top-left,.Toastify__toast-container--top-right{top:0;transform:translateX(0)}.Toastify__toast-container--bottom-center,.Toastify__toast-container--bottom-left,.Toastify__toast-container--bottom-right{bottom:0;transform:translateX(0)}.Toastify__toast-container--rtl{right:0;left:auto}}.Toastify__toast{position:relative;min-height:var(--toastify-toast-min-height);box-sizing:border-box;margin-bottom:1rem;padding:8px;border-radius:4px;box-shadow:0 1px 10px 0 rgba(0,0,0,.1),0 2px 15px 0 rgba(0,0,0,.05);display:-ms-flexbox;display:flex;-ms-flex-pack:justify;justify-content:space-between;max-height:var(--toastify-toast-max-height);overflow:hidden;font-family:var(--toastify-font-family);cursor:default;direction:ltr;z-index:0}.Toastify__toast--rtl{direction:rtl}.Toastify__toast--close-on-click{cursor:pointer}.Toastify__toast-body{margin:auto 0;-ms-flex:1 1 auto;flex:1 1 auto;padding:6px;display:-ms-flexbox;display:flex;-ms-flex-align:center;align-items:center}.Toastify__toast-body>div:last-child{word-break:break-word;-ms-flex:1;flex:1}.Toastify__toast-icon{-webkit-margin-end:10px;margin-inline-end:10px;width:20px;-ms-flex-negative:0;flex-shrink:0;display:-ms-flexbox;display:flex}.Toastify--animate{animation-fill-mode:both;animation-duration:.7s}.Toastify--animate-icon{animation-fill-mode:both;animation-duration:.3s}@media only screen and (max-width:480px){.Toastify__toast{margin-bottom:0;border-radius:0}}.Toastify__toast-theme--dark{background:var(--toastify-color-dark);color:var(--toastify-text-color-dark)}.Toastify__toast-theme--colored.Toastify__toast--default,.Toastify__toast-theme--light{background:var(--toastify-color-light);color:var(--toastify-text-color-light)}.Toastify__toast-theme--colored.Toastify__toast--info{color:var(--toastify-text-color-info);background:var(--toastify-color-info)}.Toastify__toast-theme--colored.Toastify__toast--success{color:var(--toastify-text-color-success);background:var(--toastify-color-success)}.Toastify__toast-theme--colored.Toastify__toast--warning{color:var(--toastify-text-color-warning);background:var(--toastify-color-warning)}.Toastify__toast-theme--colored.Toastify__toast--error{color:var(--toastify-text-color-error);background:var(--toastify-color-error)}.Toastify__progress-bar-theme--light{background:var(--toastify-color-progress-light)}.Toastify__progress-bar-theme--dark{background:var(--toastify-color-progress-dark)}.Toastify__progress-bar--info{background:var(--toastify-color-progress-info)}.Toastify__progress-bar--success{background:var(--toastify-color-progress-success)}.Toastify__progress-bar--warning{background:var(--toastify-color-progress-warning)}.Toastify__progress-bar--error{background:var(--toastify-color-progress-error)}.Toastify__progress-bar-theme--colored.Toastify__progress-bar--error,.Toastify__progress-bar-theme--colored.Toastify__progress-bar--info,.Toastify__progress-bar-theme--colored.Toastify__progress-bar--success,.Toastify__progress-bar-theme--colored.Toastify__progress-bar--warning{background:var(--toastify-color-transparent)}.Toastify__close-button{color:#fff;background:transparent;outline:none;border:none;padding:0;cursor:pointer;opacity:.7;transition:.3s ease;-ms-flex-item-align:start;align-self:flex-start}.Toastify__close-button--light{color:#000;opacity:.3}.Toastify__close-button>svg{fill:currentColor;height:16px;width:14px}.Toastify__close-button:focus,.Toastify__close-button:hover{opacity:1}@keyframes Toastify__trackProgress{0%{transform:scaleX(1)}to{transform:scaleX(0)}}.Toastify__progress-bar{position:absolute;bottom:0;left:0;width:100%;height:5px;z-index:var(--toastify-z-index);opacity:.7;transform-origin:left}.Toastify__progress-bar--animated{animation:Toastify__trackProgress linear 1 forwards}.Toastify__progress-bar--controlled{transition:transform .2s}.Toastify__progress-bar--rtl{right:0;left:auto;transform-origin:right}.Toastify__spinner{width:20px;height:20px;box-sizing:border-box;border:2px solid;border-radius:100%;border-color:var(--toastify-spinner-color-empty-area);border-right-color:var(--toastify-spinner-color);animation:Toastify__spin .65s linear infinite}@keyframes Toastify__bounceInRight{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(3000px,0,0)}60%{opacity:1;transform:translate3d(-25px,0,0)}75%{transform:translate3d(10px,0,0)}90%{transform:translate3d(-5px,0,0)}to{transform:none}}@keyframes Toastify__bounceOutRight{20%{opacity:1;transform:translate3d(-20px,0,0)}to{opacity:0;transform:translate3d(2000px,0,0)}}@keyframes Toastify__bounceInLeft{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(-3000px,0,0)}60%{opacity:1;transform:translate3d(25px,0,0)}75%{transform:translate3d(-10px,0,0)}90%{transform:translate3d(5px,0,0)}to{transform:none}}@keyframes Toastify__bounceOutLeft{20%{opacity:1;transform:translate3d(20px,0,0)}to{opacity:0;transform:translate3d(-2000px,0,0)}}@keyframes Toastify__bounceInUp{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(0,3000px,0)}60%{opacity:1;transform:translate3d(0,-20px,0)}75%{transform:translate3d(0,10px,0)}90%{transform:translate3d(0,-5px,0)}to{transform:translateZ(0)}}@keyframes Toastify__bounceOutUp{20%{transform:translate3d(0,-10px,0)}40%,45%{opacity:1;transform:translate3d(0,20px,0)}to{opacity:0;transform:translate3d(0,-2000px,0)}}@keyframes Toastify__bounceInDown{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(0,-3000px,0)}60%{opacity:1;transform:translate3d(0,25px,0)}75%{transform:translate3d(0,-10px,0)}90%{transform:translate3d(0,5px,0)}to{transform:none}}@keyframes Toastify__bounceOutDown{20%{transform:translate3d(0,10px,0)}40%,45%{opacity:1;transform:translate3d(0,-20px,0)}to{opacity:0;transform:translate3d(0,2000px,0)}}.Toastify__bounce-enter--bottom-left,.Toastify__bounce-enter--top-left{animation-name:Toastify__bounceInLeft}.Toastify__bounce-enter--bottom-right,.Toastify__bounce-enter--top-right{animation-name:Toastify__bounceInRight}.Toastify__bounce-enter--top-center{animation-name:Toastify__bounceInDown}.Toastify__bounce-enter--bottom-center{animation-name:Toastify__bounceInUp}.Toastify__bounce-exit--bottom-left,.Toastify__bounce-exit--top-left{animation-name:Toastify__bounceOutLeft}.Toastify__bounce-exit--bottom-right,.Toastify__bounce-exit--top-right{animation-name:Toastify__bounceOutRight}.Toastify__bounce-exit--top-center{animation-name:Toastify__bounceOutUp}.Toastify__bounce-exit--bottom-center{animation-name:Toastify__bounceOutDown}@keyframes Toastify__zoomIn{0%{opacity:0;transform:scale3d(.3,.3,.3)}50%{opacity:1}}@keyframes Toastify__zoomOut{0%{opacity:1}50%{opacity:0;transform:scale3d(.3,.3,.3)}to{opacity:0}}.Toastify__zoom-enter{animation-name:Toastify__zoomIn}.Toastify__zoom-exit{animation-name:Toastify__zoomOut}@keyframes Toastify__flipIn{0%{transform:perspective(400px) rotateX(90deg);animation-timing-function:ease-in;opacity:0}40%{transform:perspective(400px) rotateX(-20deg);animation-timing-function:ease-in}60%{transform:perspective(400px) rotateX(10deg);opacity:1}80%{transform:perspective(400px) rotateX(-5deg)}to{transform:perspective(400px)}}@keyframes Toastify__flipOut{0%{transform:perspective(400px)}30%{transform:perspective(400px) rotateX(-20deg);opacity:1}to{transform:perspective(400px) rotateX(90deg);opacity:0}}.Toastify__flip-enter{animation-name:Toastify__flipIn}.Toastify__flip-exit{animation-name:Toastify__flipOut}@keyframes Toastify__slideInRight{0%{transform:translate3d(110%,0,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideInLeft{0%{transform:translate3d(-110%,0,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideInUp{0%{transform:translate3d(0,110%,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideInDown{0%{transform:translate3d(0,-110%,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideOutRight{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(110%,0,0)}}@keyframes Toastify__slideOutLeft{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(-110%,0,0)}}@keyframes Toastify__slideOutDown{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(0,500px,0)}}@keyframes Toastify__slideOutUp{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(0,-500px,0)}}.Toastify__slide-enter--bottom-left,.Toastify__slide-enter--top-left{animation-name:Toastify__slideInLeft}.Toastify__slide-enter--bottom-right,.Toastify__slide-enter--top-right{animation-name:Toastify__slideInRight}.Toastify__slide-enter--top-center{animation-name:Toastify__slideInDown}.Toastify__slide-enter--bottom-center{animation-name:Toastify__slideInUp}.Toastify__slide-exit--bottom-left,.Toastify__slide-exit--top-left{animation-name:Toastify__slideOutLeft}.Toastify__slide-exit--bottom-right,.Toastify__slide-exit--top-right{animation-name:Toastify__slideOutRight}.Toastify__slide-exit--top-center{animation-name:Toastify__slideOutUp}.Toastify__slide-exit--bottom-center{animation-name:Toastify__slideOutDown}@keyframes Toastify__spin{0%{transform:rotate(0deg)}to{transform:rotate(1turn)}}</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

:root {
  --toastify-color-light: var(--jp-layout-color1);
  --toastify-color-dark: var(--jp-layout-color1);
  --toastify-color-info: var(--jp-info-color1);
  --toastify-color-success: var(--jp-success-color1);
  --toastify-color-warning: var(--jp-warn-color1);
  --toastify-color-error: var(--jp-error-color1);
  --toastify-color-transparent: rgba(255, 255, 255, 0.7);
  --toastify-icon-color-info: var(--toastify-color-info);
  --toastify-icon-color-success: var(--toastify-color-success);
  --toastify-icon-color-warning: var(--toastify-color-warning);
  --toastify-icon-color-error: var(--toastify-color-error);
  --toastify-toast-width: 25em;
  --toastify-toast-background: var(--jp-layout-color1);
  --toastify-toast-min-height: 64px;
  --toastify-toast-max-height: 800px;
  --toastify-font-family: var(--jp-ui-font-family);
  --toastify-z-index: 9999;
  --toastify-text-color-light: var(--jp-ui-font-color1);
  --toastify-text-color-dark: var(--jp-ui-font-color1);
  --toastify-text-color-info: var(--jp-ui-font-color1);
  --toastify-text-color-success: var(--jp-ui-font-color1);
  --toastify-text-color-warning: var(--jp-ui-font-color1);
  --toastify-text-color-error: var(--jp-ui-font-color1);
  --toastify-spinner-color: #616161;
  --toastify-spinner-color-empty-area: #e0e0e0;
  --toastify-color-progress-light: linear-gradient(
    to right,
    #4cd964,
    #5ac8fa,
    #007aff,
    #34aadc,
    #5856d6,
    #ff2d55
  );
  --toastify-color-progress-dark: #bb86fc;
  --toastify-color-progress-info: var(--toastify-color-info);
  --toastify-color-progress-success: var(--toastify-color-success);
  --toastify-color-progress-warning: var(--toastify-color-warning);
  --toastify-color-progress-error: var(--toastify-color-error);
}

.jp-Notification-List {
  list-style: none;
  margin: 0;
  padding: 4px;
  width: var(--toastify-toast-width);
  overflow-y: auto;
  max-height: 55vh;
  box-sizing: border-box;
  background-color: var(--jp-layout-color2);
}

.jp-Notification-Header {
  padding: 0 4px;
  margin: 0;
  align-items: center;
  user-select: none;
}

.jp-Notification-List-Item {
  padding: 2px 0;
}

.jp-Notification-List .Toastify__toast {
  margin: 0;
}

.jp-Notification-Status.jp-mod-selected {
  background-color: var(--jp-brand-color1);
}

.jp-Notification-Status.jp-mod-selected .jp-Notification-Status-Text {
  color: var(--jp-ui-inverse-font-color1);
}

.Toastify__toast {
  min-height: unset;
  padding: 4px;
  font-size: var(--jp-ui-font-size1);
  border-width: var(--jp-border-width);
  border-radius: var(--jp-border-radius);
  border-color: var(--jp-border-color1);
  box-shadow: var(--jp-elevation-z4);
  cursor: default;
}

.Toastify__toast-body {
  display: flex;
  flex-grow: 1;
}

.jp-Notification-Toast-Close {
  padding: 0;
  position: absolute;
  right: 0.1px;
  cursor: pointer;
}

.jp-Notification-Toast-Close-Margin {
  margin-right: 4px;
}

.jp-toastContainer .jp-Notification-Toast-Close:hover {
  /* The close button has its own hover style */
  background: none;
}

.Toastify__toast.jp-Notification-Toast-error {
  border-top: 5px solid var(--jp-error-color1);
}

.Toastify__toast.jp-Notification-Toast-warning {
  border-top: 5px solid var(--jp-warn-color1);
}

.Toastify__toast.jp-Notification-Toast-info {
  border-top: 5px solid var(--jp-info-color1);
}

.Toastify__toast.jp-Notification-Toast-success {
  border-top: 5px solid var(--jp-success-color1);
}

.Toastify__toast.jp-Notification-Toast-in-progress {
  border-top: 5px solid var(--jp-layout-color1);
}

.Toastify__toast-body a {
  color: var(--jp-content-link-color);
}

.Toastify__toast-body a:hover {
  color: var(--jp-content-link-color);
  text-decoration: underline;
}

.jp-toast-message {
  padding-inline-end: 16px;
}

/* p elements are added by the markdown rendering.
 * Removing its default margin allows to reduce toast size.
 */
.Toastify__toast-body p:first-child,
.Toastify__toast-body h1:first-child,
.Toastify__toast-body h2:first-child,
.Toastify__toast-body h3:first-child,
.Toastify__toast-body h4:first-child,
.Toastify__toast-body h5:first-child,
.Toastify__toast-body h6:first-child,
.Toastify__toast-body ol:first-child,
.Toastify__toast-body ul:first-child {
  margin-top: 0;
}

.Toastify__toast-body p:last-child,
.Toastify__toast-body h1:last-child,
.Toastify__toast-body h2:last-child,
.Toastify__toast-body h3:last-child,
.Toastify__toast-body h4:last-child,
.Toastify__toast-body h5:last-child,
.Toastify__toast-body h6:last-child,
.Toastify__toast-body ol:last-child,
.Toastify__toast-body ul:last-child {
  margin-bottom: 0;
}

.jp-toast-buttonBar {
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  flex: 0 0 auto;
  padding-block-start: 8px;
}

.jp-toast-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-toast-button {
  margin-top: 1px;
  margin-bottom: 1px;
  margin-right: 0;
  margin-left: 3px;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color2);
  border: none;
}

.jp-toast-button:focus {
  outline: 1px solid var(--jp-reject-color-normal, var(--jp-layout-color2));
  outline-offset: 1px;
  -moz-outline-radius: 0;
}

.jp-toast-button:focus-visible {
  border: none;
}

.jp-toast-button:hover {
  background-color: var(--jp-layout-color3);
}

.jp-toast-button.jp-mod-accept {
  background: var(--jp-accept-color-normal, var(--jp-brand-color1));
  color: var(--jp-ui-inverse-font-color1);
}

.jp-toast-button.jp-mod-accept:focus {
  outline-color: var(--jp-accept-color-normal, var(--jp-brand-color1));
}

.jp-toast-button.jp-mod-accept:hover {
  background: var(--jp-accept-color-hover, var(--jp-brand-color0));
}

.jp-toast-button.jp-mod-warn {
  background: var(--jp-warn-color-normal, var(--jp-warn-color1));
  color: var(--jp-ui-inverse-font-color1);
}

.jp-toast-button.jp-mod-warn:focus {
  outline-color: var(--jp-warn-color-normal, var(--jp-warn-color1));
}

.jp-toast-button.jp-mod-warn:hover {
  background: var(--jp-warn-color-hover, var(--jp-warn-color0));
}

.jp-toast-button.jp-mod-link {
  color: var(--jp-content-link-color);
  text-decoration: underline;
  text-decoration-color: var(--jp-content-link-color);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

#jupyterlab-splash {
  z-index: 10;
  position: absolute;
  overflow: hidden;
  width: 100%;
  height: 100%;
  background-position: center 40%;
  background-repeat: no-repeat;
  background-size: cover;
}

#jupyterlab-splash.light {
  background-color: white;
}

#jupyterlab-splash.dark {
  background-color: var(--md-grey-900);
}

.splash-fade {
  animation: 0.5s fade-out forwards;
}

#galaxy {
  position: relative;
  width: 100%;
  height: 100%;
}

.planet {
  background-repeat: no-repeat;
  background-size: cover;
  animation-iteration-count: infinite;
  animation-name: orbit;
}

#moon1.orbit {
  opacity: 1;
  animation: orbit 2s ease;
  width: 200px;
  height: 140px;
  margin-top: -53px;
  margin-left: -54px;
}

#moon2.orbit {
  opacity: 1;
  animation: orbit 2s ease;
  width: 132px;
  height: 180px;
  margin-top: -66px;
  margin-left: -85px;
}

#moon3.orbit {
  opacity: 1;
  display: flex;
  align-items: flex-end;
  animation: orbit 2s ease;
  width: 220px;
  height: 166px;
  margin-top: -96px;
  margin-left: -50px;
}

#moon1 .planet {
  height: 12px;
  width: 12px;
  border-radius: 50%;
}

#moon2 .planet {
  height: 16px;
  width: 16px;
  border-radius: 50%;
  float: right;
}

#moon3 .planet {
  height: 20px;
  width: 20px;
  border-radius: 50%;
}

#jupyterlab-splash.light #moon1 .planet {
  background-color: #6f7070;
}

#jupyterlab-splash.light #moon2 .planet {
  background-color: #767677;
}

#jupyterlab-splash.light #moon3 .planet {
  background-color: #989798;
}

#jupyterlab-splash.dark #moon1 .planet,
#jupyterlab-splash.dark #moon2 .planet,
#jupyterlab-splash.dark #moon3 .planet {
  background-color: white;
}

.orbit {
  animation-iteration-count: 1;
  position: absolute;
  top: 50%;
  left: 50%;
  border-radius: 50%;
}

@keyframes orbit {
  0% {
    transform: rotateZ(0deg);
  }

  100% {
    transform: rotateZ(-720deg);
  }
}

@keyframes orbit2 {
  0% {
    transform: rotateZ(0deg);
  }

  100% {
    transform: rotateZ(720deg);
  }
}

@keyframes fade-in {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}

@keyframes fade-out {
  0% {
    opacity: 1;
  }

  100% {
    opacity: 0;
  }
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

:root {
  --jp-private-shortcuts-key-padding-horizontal: 0.47em;
  --jp-private-shortcuts-key-padding-vertical: 0.28em;
  --jp-private-shortcuts-label-padding-horizontal: 0.47em;
}

.jp-ContextualShortcut-TableRow {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
}

.jp-ContextualShortcut-TableItem {
  margin-left: auto;
  margin-right: auto;
  color: var(--jp-inverse-layout-color0);
  font-size: var(--jp-ui-font-size1);
  line-height: 2em;
  padding-right: var(--jp-private-shortcuts-label-padding-horizontal);
}

.jp-ContextualShortcut-TableLastRow {
  height: 2em;
}

.jp-ContextualShortcut-Key {
  font-family: var(--jp-code-font-family);
  border-width: var(--jp-border-width);
  border-radius: var(--jp-border-radius);
  border-style: solid;
  border-color: var(--jp-border-color1);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  padding-left: var(--jp-private-shortcuts-key-padding-horizontal);
  padding-right: var(--jp-private-shortcuts-key-padding-horizontal);
  padding-top: var(--jp-private-shortcuts-key-padding-vertical);
  padding-bottom: var(--jp-private-shortcuts-key-padding-vertical);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0;
  padding: 0;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}

.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}

.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}

.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}

.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}

.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}

.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}

.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}

.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}

.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}

.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}

.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}

.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}

.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}

.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}

.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}

.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);

  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

/* stylelint-disable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0;
}

/* stylelint-enable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  table-layout: fixed;
  margin-left: auto;
  margin-bottom: 1em;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}

[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}

.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}

.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}

.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}

.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}

.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}

.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}

.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}

.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: var(--jp-ui-font-size0);
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-lineFormSearch {
  padding: 4px 12px;
  background-color: var(--jp-layout-color2);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
  font-size: var(--jp-ui-font-size1);
}

.jp-lineFormCaption {
  font-size: var(--jp-ui-font-size0);
  line-height: var(--jp-ui-font-size1);
  margin-top: 4px;
  color: var(--jp-ui-font-color0);
}

.jp-baseLineForm {
  border: none;
  border-radius: 0;
  position: absolute;
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  outline: none;
}

.jp-lineFormButtonContainer {
  top: 4px;
  right: 8px;
  height: 24px;
  padding: 0 12px;
  width: 12px;
}

.jp-lineFormButtonIcon {
  top: 0;
  right: 0;
  background-color: var(--jp-brand-color1);
  height: 100%;
  width: 100%;
  box-sizing: border-box;
  padding: 4px 6px;
}

.jp-lineFormButton {
  top: 0;
  right: 0;
  background-color: transparent;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
}

.jp-lineFormWrapper {
  overflow: hidden;
  padding: 0 8px;
  border: 1px solid var(--jp-border-color0);
  background-color: var(--jp-input-active-background);
  height: 22px;
}

.jp-lineFormWrapperFocusWithin {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-lineFormInput {
  background: transparent;
  width: 200px;
  height: 100%;
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  line-height: 28px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
.jp-DocumentSearch-input {
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  font-size: var(--jp-ui-font-size1);
  background-color: var(--jp-layout-color0);
  font-family: var(--jp-ui-font-family);
  padding: 2px 1px;
  resize: none;
  white-space: pre;
}

.jp-DocumentSearch-overlay {
  position: absolute;
  background-color: var(--jp-toolbar-background);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  border-left: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  top: 0;
  right: 0;
  z-index: 7;
  min-width: 405px;
  padding: 2px;
  font-size: var(--jp-ui-font-size1);

  --jp-private-document-search-button-height: 20px;
}

.jp-DocumentSearch-overlay button {
  background-color: var(--jp-toolbar-background);
  outline: 0;
}

.jp-DocumentSearch-overlay button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-overlay button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-overlay-row {
  display: flex;
  align-items: center;
  margin-bottom: 2px;
}

.jp-DocumentSearch-button-content {
  display: inline-block;
  cursor: pointer;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-button-content svg {
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-input-wrapper {
  border: var(--jp-border-width) solid var(--jp-border-color0);
  display: flex;
  background-color: var(--jp-layout-color0);
  margin: 2px;
}

.jp-DocumentSearch-input-wrapper:focus-within {
  border-color: var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper {
  all: initial;
  overflow: hidden;
  display: inline-block;
  border: none;
  box-sizing: border-box;
}

.jp-DocumentSearch-toggle-wrapper {
  flex-shrink: 0;
  width: 14px;
  height: 14px;
}

.jp-DocumentSearch-button-wrapper {
  flex-shrink: 0;
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
}

.jp-DocumentSearch-toggle-wrapper:focus,
.jp-DocumentSearch-button-wrapper:focus {
  outline: var(--jp-border-width) solid
    var(--jp-cell-editor-active-border-color);
  outline-offset: -1px;
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper,
.jp-DocumentSearch-button-content:focus {
  outline: none;
}

.jp-DocumentSearch-toggle-placeholder {
  width: 5px;
}

.jp-DocumentSearch-input-button::before {
  display: block;
  padding-top: 100%;
}

.jp-DocumentSearch-input-button-off {
  opacity: var(--jp-search-toggle-off-opacity);
}

.jp-DocumentSearch-input-button-off:hover {
  opacity: var(--jp-search-toggle-hover-opacity);
}

.jp-DocumentSearch-input-button-on {
  opacity: var(--jp-search-toggle-on-opacity);
}

.jp-DocumentSearch-index-counter {
  padding-left: 10px;
  padding-right: 10px;
  user-select: none;
  min-width: 35px;
  display: inline-block;
}

.jp-DocumentSearch-up-down-wrapper {
  display: inline-block;
  padding-right: 2px;
  margin-left: auto;
  white-space: nowrap;
}

.jp-DocumentSearch-spacer {
  margin-left: auto;
}

.jp-DocumentSearch-up-down-wrapper button {
  outline: 0;
  border: none;
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
  vertical-align: middle;
  margin: 1px 5px 2px;
}

.jp-DocumentSearch-up-down-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-up-down-button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-filter-button {
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-filter-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled:hover {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-search-options {
  padding: 0 8px;
  margin-left: 3px;
  width: 100%;
  display: grid;
  justify-content: start;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  justify-items: stretch;
}

.jp-DocumentSearch-search-filter-disabled {
  color: var(--jp-ui-font-color2);
}

.jp-DocumentSearch-search-filter {
  display: flex;
  align-items: center;
  user-select: none;
}

.jp-DocumentSearch-regex-error {
  color: var(--jp-error-color0);
}

.jp-DocumentSearch-replace-button-wrapper {
  overflow: hidden;
  display: inline-block;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color0);
  margin: auto 2px;
  padding: 1px 4px;
  height: calc(var(--jp-private-document-search-button-height) + 2px);
  flex-shrink: 0;
}

.jp-DocumentSearch-replace-button-wrapper:focus {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-replace-button {
  display: inline-block;
  text-align: center;
  cursor: pointer;
  box-sizing: border-box;
  color: var(--jp-ui-font-color1);

  /* height - 2 * (padding of wrapper) */
  line-height: calc(var(--jp-private-document-search-button-height) - 2px);
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-replace-button:focus {
  outline: none;
}

.jp-DocumentSearch-replace-wrapper-class {
  margin-left: 14px;
  display: flex;
}

.jp-DocumentSearch-replace-toggle {
  border: none;
  background-color: var(--jp-toolbar-background);
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-replace-toggle:hover {
  background-color: var(--jp-layout-color2);
}

/*
  The following few rules allow the search box to expand horizontally,
  as the text within it grows. This is done by using putting
  the text within a wrapper element and using that wrapper for sizing,
  as <textarea> and <input> tags do not grow automatically.
  This is the underlying technique:
  https://til.simonwillison.net/css/resizing-textarea
*/
.jp-DocumentSearch-input-label::after {
  content: attr(data-value) ' ';
  visibility: hidden;
  white-space: pre;
}

.jp-DocumentSearch-input-label {
  display: inline-grid;
  align-items: stretch;
}

.jp-DocumentSearch-input-label::after,
.jp-DocumentSearch-input-label > .jp-DocumentSearch-input {
  width: auto;
  min-width: 1em;
  grid-area: 1/2;
  font: inherit;
  padding: 2px 3px;
  margin: 0;
  resize: none;
  background: none;
  appearance: none;
  border: none;
  overflow: hidden;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.cm-editor {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;

  /* Changed to auto to autogrow */
}

.cm-editor pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .cm-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

.jp-CodeMirrorEditor {
  cursor: text;
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (width >= 2138px) and (width <= 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (width >= 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.cm-editor.jp-mod-readOnly .cm-cursor {
  display: none;
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.cm-searching,
.cm-searching span {
  /* `.cm-searching span`: we need to override syntax highlighting */
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.cm-searching::selection,
.cm-searching span::selection {
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.jp-current-match > .cm-searching,
.jp-current-match > .cm-searching span,
.cm-searching > .jp-current-match,
.cm-searching > .jp-current-match span {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.jp-current-match > .cm-searching::selection,
.cm-searching > .jp-current-match::selection,
.jp-current-match > .cm-searching span::selection {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.cm-trailingspace {
  background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAsElEQVQIHQGlAFr/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7+r3zKmT0/+pk9P/7+r3zAAAAAAAAAAABAAAAAAAAAAA6OPzM+/q9wAAAAAA6OPzMwAAAAAAAAAAAgAAAAAAAAAAGR8NiRQaCgAZIA0AGR8NiQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQyoYJ/SY80UAAAAASUVORK5CYII=);
  background-position: center left;
  background-repeat: repeat-x;
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .cm-ySelectionCaret {
  position: relative;
  border-left: 1px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret > .cm-ySelectionInfo {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -1px;
  font-size: 0.95em;
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 101;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .cm-ySelectionInfo {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret:hover > .cm-ySelectionInfo {
  opacity: 1;
  transition-delay: 0s;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: flex;
  flex-direction: row;
  width: 100%;
  overflow: hidden;
}

.jp-OutputPrompt {
  width: var(--jp-cell-prompt-width);
  flex: 0 0 var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-OutputArea-output {
  width: 100%;
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea .jp-RenderedText {
  padding-left: 1ch;
}

/**
 * Prompt overlay.
 */

.jp-OutputArea-promptOverlay {
  position: absolute;
  top: 0;
  width: var(--jp-cell-prompt-width);
  height: 100%;
  opacity: 0.5;
}

.jp-OutputArea-promptOverlay:hover {
  background: var(--jp-layout-color2);
  box-shadow: inset 0 0 1px var(--jp-inverse-layout-color0);
  cursor: zoom-out;
}

.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay:hover {
  cursor: zoom-in;
}

.jp-OutputArea-child .jp-OutputArea-output {
  flex-grow: 1;
  flex-shrink: 1;
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0;
  padding: 0;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

.jp-TrimmedOutputs pre {
  background: var(--jp-layout-color3);
  font-size: calc(var(--jp-code-font-size) * 1.4);
  text-align: center;
  text-transform: uppercase;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/* Hide empty lines in the output area, for instance due to cleared widgets */
.jp-OutputArea-prompt:empty {
  padding: 0;
  border: 0;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0;
  width: 100%;
  flex: 1 1 auto;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;

  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;

  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0 0.25em;
  margin: 0 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input::placeholder {
  opacity: 0;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

.jp-Stdin-input:focus::placeholder {
  opacity: 1;
}

.jp-OutputArea-stdin-hiding {
  /* soft-hide the output, preserving focus */
  opacity: 0;
  height: 0;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

@media print {
  .jp-OutputArea-child {
    display: table;
    table-layout: fixed;
    break-inside: avoid-page;
  }

  .jp-OutputArea-prompt {
    display: table-cell;
    vertical-align: top;
  }

  .jp-OutputArea-output {
    display: table-cell;
  }
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (width <= 760px) {
  .jp-OutputArea-child {
    flex-direction: column;
  }

  .jp-OutputPrompt {
    flex: 0 0 auto;
    text-align: left;
  }

  .jp-OutputArea-promptOverlay {
    display: none;
  }
}

/* Trimmed outputs warning */
.jp-TrimmedOutputs > a {
  margin: 10px;
  text-decoration: none;
  cursor: pointer;
}

.jp-TrimmedOutputs > a:hover {
  text-decoration: none;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Table of Contents
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toc-active-width: 4px;
}

.jp-TableOfContents {
  display: flex;
  flex-direction: column;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  height: 100%;
}

.jp-TableOfContents-placeholder {
  text-align: center;
}

.jp-TableOfContents-placeholderContent {
  color: var(--jp-content-font-color2);
  padding: 8px;
}

.jp-TableOfContents-placeholderContent > h3 {
  margin-bottom: var(--jp-content-heading-margin-bottom);
}

.jp-TableOfContents .jp-SidePanel-content {
  overflow-y: auto;
}

.jp-TableOfContents-tree {
  margin: 4px;
}

.jp-TableOfContents ol {
  list-style-type: none;
}

/* stylelint-disable-next-line selector-max-type */
.jp-TableOfContents li > ol {
  /* Align left border with triangle icon center */
  padding-left: 11px;
}

.jp-TableOfContents-content {
  /* left margin for the active heading indicator */
  margin: 0 0 0 var(--jp-private-toc-active-width);
  padding: 0;
  background-color: var(--jp-layout-color1);
}

.jp-tocItem {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-tocItem-heading {
  display: flex;
  cursor: pointer;
}

.jp-tocItem-heading:hover {
  background-color: var(--jp-layout-color2);
}

.jp-tocItem-content {
  display: block;
  padding: 4px 0;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow-x: hidden;
}

.jp-tocItem-collapser {
  height: 20px;
  margin: 2px 2px 0;
  padding: 0;
  background: none;
  border: none;
  cursor: pointer;
}

.jp-tocItem-collapser:hover {
  background-color: var(--jp-layout-color3);
}

/* Active heading indicator */

.jp-tocItem-heading::before {
  content: ' ';
  background: transparent;
  width: var(--jp-private-toc-active-width);
  height: 24px;
  position: absolute;
  left: 0;
  border-radius: var(--jp-border-radius);
}

.jp-tocItem-heading.jp-tocItem-active::before {
  background-color: var(--jp-brand-color1);
}

.jp-tocItem-heading:hover.jp-tocItem-active::before {
  background: var(--jp-brand-color0);
  opacity: 1;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;

  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0;
  bottom: 0;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Hiding collapsers in print mode.

Note: input and output wrappers have "display: block" propery in print mode.
*/

@media print {
  .jp-Collapser {
    display: none;
  }
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0;
  width: 100%;
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: flex;
  flex-direction: row;
  width: 100%;
  overflow: hidden;
}

.jp-InputArea-editor {
  flex: 1 1 auto;
  overflow: hidden;

  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  background: var(--jp-cell-editor-background);
}

.jp-InputPrompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/*-----------------------------------------------------------------------------
| Print
|----------------------------------------------------------------------------*/
@media print {
  .jp-InputArea {
    display: table;
    table-layout: fixed;
  }

  .jp-InputArea-editor {
    display: table-cell;
    vertical-align: top;
  }

  .jp-InputPrompt {
    display: table-cell;
    vertical-align: top;
  }
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (width <= 760px) {
  .jp-InputArea {
    flex-direction: column;
  }

  .jp-InputArea-editor {
    margin-left: var(--jp-code-padding);
  }

  .jp-InputPrompt {
    flex: 0 0 auto;
    text-align: left;
  }
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: flex;
  flex-direction: row;
  width: 100%;
}

.jp-Placeholder-prompt {
  flex: 0 0 var(--jp-cell-prompt-width);
  box-sizing: border-box;
}

.jp-Placeholder-content {
  flex: 1 1 auto;
  padding: 4px 6px;
  border: 1px solid transparent;
  border-radius: 0;
  background: none;
  box-sizing: border-box;
  cursor: pointer;
}

.jp-Placeholder-contentContainer {
  display: flex;
}

.jp-Placeholder-content:hover,
.jp-InputPlaceholder > .jp-Placeholder-content:hover {
  border-color: var(--jp-layout-color3);
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

.jp-PlaceholderText {
  white-space: nowrap;
  overflow-x: hidden;
  color: var(--jp-inverse-layout-color3);
  font-family: var(--jp-code-font-family);
}

.jp-InputPlaceholder > .jp-Placeholder-content {
  border-color: var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
}

/*-----------------------------------------------------------------------------
| Print
|----------------------------------------------------------------------------*/
@media print {
  .jp-Placeholder {
    display: table;
    table-layout: fixed;
  }

  .jp-Placeholder-content {
    display: table-cell;
  }

  .jp-Placeholder-prompt {
    display: table-cell;
  }
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0;
  margin: 0;

  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 24em;
  margin-left: var(--jp-private-cell-scrolling-output-offset);
  resize: vertical;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea[style*='height'] {
  max-height: unset;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea::after {
  content: ' ';
  box-shadow: inset 0 0 6px 2px rgb(0 0 0 / 30%);
  width: 100%;
  height: 100%;
  position: sticky;
  bottom: 0;
  top: 0;
  margin-top: -50%;
  float: left;
  display: block;
  pointer-events: none;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-child {
  padding-top: 6px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
  flex: 0 0
    calc(
      var(--jp-cell-prompt-width) -
        var(--jp-private-cell-scrolling-output-offset)
    );
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay {
  left: calc(-1 * var(--jp-private-cell-scrolling-output-offset));
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  flex: 1 1 auto;
  width: 100%;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

/* collapseHeadingButton (show always if hiddenCellsButton is _not_ shown) */
.jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  font-size: var(--jp-code-font-size);
  position: absolute;
  background-color: transparent;
  background-size: 25px;
  background-repeat: no-repeat;
  background-position-x: center;
  background-position-y: top;
  background-image: var(--jp-icon-caret-down);
  right: 0;
  top: 0;
  bottom: 0;
}

.jp-collapseHeadingButton.jp-mod-collapsed {
  background-image: var(--jp-icon-caret-right);
}

/*
 set the container font size to match that of content
 so that the nested collapse buttons have the right size
*/
.jp-MarkdownCell .jp-InputPrompt {
  font-size: var(--jp-content-font-size1);
}

/*
  Align collapseHeadingButton with cell top header
  The font sizes are identical to the ones in packages/rendermime/style/base.css
*/
.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='1'] {
  font-size: var(--jp-content-font-size5);
  background-position-y: calc(0.3 * var(--jp-content-font-size5));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='2'] {
  font-size: var(--jp-content-font-size4);
  background-position-y: calc(0.3 * var(--jp-content-font-size4));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='3'] {
  font-size: var(--jp-content-font-size3);
  background-position-y: calc(0.3 * var(--jp-content-font-size3));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='4'] {
  font-size: var(--jp-content-font-size2);
  background-position-y: calc(0.3 * var(--jp-content-font-size2));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='5'] {
  font-size: var(--jp-content-font-size1);
  background-position-y: top;
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='6'] {
  font-size: var(--jp-content-font-size0);
  background-position-y: top;
}

/* collapseHeadingButton (show only on (hover,active) if hiddenCellsButton is shown) */
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-collapseHeadingButton {
  display: none;
}

.jp-Notebook.jp-mod-showHiddenCellsButton
  :is(.jp-MarkdownCell:hover, .jp-mod-active)
  .jp-collapseHeadingButton {
  display: flex;
}

/* showHiddenCellsButton (only show if jp-mod-showHiddenCellsButton is set, which
is a consequence of the showHiddenCellsButton option in Notebook Settings)*/
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
  display: flex;
}

.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-showHiddenCellsButton {
  display: none;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Using block instead of flex to allow the use of the break-inside CSS property for
cell outputs.
*/

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }

  .jp-MarkdownOutput {
    display: table-cell;
  }
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-notebook-toolbar-padding: 2px 5px 2px 2px;
}

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: var(--jp-notebook-toolbar-padding);

  /* disable paint containment from lumino 2.0 default strict CSS containment */
  contain: style size !important;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

.jp-Toolbar-responsive-popup {
  position: absolute;
  height: fit-content;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: flex-end;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: var(--jp-notebook-toolbar-padding);
  z-index: 1;
  right: 0;
  top: 0;
}

.jp-Toolbar > .jp-Toolbar-responsive-opener {
  margin-left: auto;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-Notebook-ExecutionIndicator {
  position: relative;
  display: inline-block;
  height: 100%;
  z-index: 9997;
}

.jp-Notebook-ExecutionIndicator-tooltip {
  visibility: hidden;
  height: auto;
  width: max-content;
  width: -moz-max-content;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color1);
  text-align: justify;
  border-radius: 6px;
  padding: 0 5px;
  position: fixed;
  display: table;
}

.jp-Notebook-ExecutionIndicator-tooltip.up {
  transform: translateX(-50%) translateY(-100%) translateY(-32px);
}

.jp-Notebook-ExecutionIndicator-tooltip.down {
  transform: translateX(calc(-100% + 16px)) translateY(5px);
}

.jp-Notebook-ExecutionIndicator-tooltip.hidden {
  display: none;
}

.jp-Notebook-ExecutionIndicator:hover .jp-Notebook-ExecutionIndicator-tooltip {
  visibility: visible;
}

.jp-Notebook-ExecutionIndicator span {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  color: var(--jp-ui-font-color1);
  line-height: 24px;
  display: block;
}

.jp-Notebook-ExecutionIndicator-progress-bar {
  display: flex;
  justify-content: center;
  height: 100%;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
 * Execution indicator
 */
.jp-tocItem-content::after {
  content: '';

  /* Must be identical to form a circle */
  width: 12px;
  height: 12px;
  background: none;
  border: none;
  position: absolute;
  right: 0;
}

.jp-tocItem-content[data-running='0']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background: none;
}

.jp-tocItem-content[data-running='1']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background-color: var(--jp-inverse-layout-color3);
}

.jp-tocItem-content[data-running='0'],
.jp-tocItem-content[data-running='1'] {
  margin-right: 12px;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Notebook-footer {
  height: 27px;
  margin-left: calc(
    var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
      var(--jp-cell-padding)
  );
  width: calc(
    100% -
      (
        var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
          var(--jp-cell-padding) + var(--jp-cell-padding)
      )
  );
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  color: var(--jp-ui-font-color3);
  margin-top: 6px;
  background: none;
  cursor: pointer;
}

.jp-Notebook-footer:focus {
  border-color: var(--jp-cell-editor-active-border-color);
}

/* For devices that support hovering, hide footer until hover */
@media (hover: hover) {
  .jp-Notebook-footer {
    opacity: 0;
  }

  .jp-Notebook-footer:focus,
  .jp-Notebook-footer:hover {
    opacity: 1;
  }
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-side-by-side-output-size: 1fr;
  --jp-side-by-side-resized-cell: var(--jp-side-by-side-output-size);
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

/* stylelint-disable selector-max-class */

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
  float: left;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}

.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt::before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: block;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-ActiveCellTool {
  padding: 12px 0;
  display: flex;
}

.jp-ActiveCellTool-Content {
  flex: 1 1 auto;
}

.jp-ActiveCellTool .jp-ActiveCellTool-CellContent {
  background: var(--jp-cell-editor-background);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  min-height: 29px;
}

.jp-ActiveCellTool .jp-InputPrompt {
  min-width: calc(var(--jp-cell-prompt-width) * 0.75);
}

.jp-ActiveCellTool-CellContent > pre {
  padding: 5px 4px;
  margin: 0;
  white-space: normal;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label,
.jp-NumberSetter label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0;
}

.jp-NumberSetter input {
  width: 100%;
  margin-top: 4px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Side-by-side Mode (.jp-mod-sideBySide)
|----------------------------------------------------------------------------*/
.jp-mod-sideBySide.jp-Notebook .jp-Notebook-cell {
  margin-top: 3em;
  margin-bottom: 3em;
  margin-left: 5%;
  margin-right: 5%;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell {
  display: grid;
  grid-template-columns: minmax(70px, 1fr) min-content minmax(
      70px,
      var(--jp-side-by-side-output-size)
    );
  grid-template-rows: auto minmax(0, 1fr) auto;
  grid-template-areas:
    'header header header'
    'input handle output'
    'footer footer footer';
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell.jp-mod-resizedCell {
  grid-template-columns: minmax(70px, 1fr) min-content minmax(
      70px,
      var(--jp-side-by-side-resized-cell)
    );
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellHeader {
  grid-area: header;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-inputWrapper {
  grid-area: input;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-outputWrapper {
  /* overwrite the default margin (no vertical separation needed in side by side move */
  margin-top: 0;
  grid-area: output;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellFooter {
  grid-area: footer;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle {
  grid-area: handle;
  user-select: none;
  display: block;
  height: 100%;
  cursor: ew-resize;
  padding: 0 var(--jp-cell-padding);
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle::after {
  content: '';
  display: block;
  background: var(--jp-border-color2);
  height: 100%;
  width: 5px;
}

.jp-mod-sideBySide.jp-Notebook
  .jp-CodeCell.jp-mod-resizedCell
  .jp-CellResizeHandle::after {
  background: var(--jp-border-color0);
}

.jp-CellResizeHandle {
  display: none;
}

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/
@media print {
  .jp-Notebook .jp-Cell .jp-InputPrompt {
    float: none;
  }
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-cell-button .jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-cell-button:hover .jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-toolbar-overlap .jp-cell-toolbar {
  display: none;
}

.jp-cell-toolbar {
  display: flex;
  flex-direction: row;
  padding: 2px 0;
  min-height: 25px;
  z-index: 6;
  position: absolute;
  top: 5px;
  right: 8px;

  /* Override .jp-Toolbar */
  background-color: transparent;
  border-bottom: inherit;
  box-shadow: inherit;
}

/* Overrides for mobile view hiding cell toolbar */
@media only screen and (width <= 760px) {
  .jp-cell-toolbar {
    display: none;
  }
}

.jp-cell-toolbar button.jp-ToolbarButtonComponent {
  cursor: pointer;
}

.jp-cell-toolbar .jp-ToolbarButton button {
  display: none;
}

.jp-cell-toolbar .jp-ToolbarButton .jp-cell-all,
.jp-CodeCell .jp-ToolbarButton .jp-cell-code,
.jp-MarkdownCell .jp-ToolbarButton .jp-cell-markdown,
.jp-RawCell .jp-ToolbarButton .jp-cell-raw {
  display: block;
}

.jp-cell-toolbar .jp-Toolbar-spacer {
  flex: 1 1 auto;
}

.jp-cell-mod-click {
  cursor: pointer;
}

/* Custom styling for rendered markdown cells so that cell toolbar is visible */
.jp-MarkdownOutput {
  border-width: var(--jp-border-width);
  border-color: transparent;
  border-style: solid;
}

.jp-mod-active .jp-MarkdownOutput {
  border-color: var(--jp-cell-editor-border-color);
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

:root {
  --jp-add-tag-extra-width: 8px;
}

.jp-CellTags-Tag {
  height: 20px;
  border-radius: 10px;
  margin-right: 5px;
  margin-bottom: 10px;
  padding: 0 8px;
  font-size: var(--jp-ui-font-size1);
  display: inline-flex;
  justify-content: center;
  align-items: center;
  max-width: calc(100% - 25px);
  border: 1px solid var(--jp-border-color1);
  color: var(--jp-ui-font-color1);
  -webkit-touch-callout: none;
  -webkit-user-select: none;
  -khtml-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-CellTags-Unapplied {
  background-color: var(--jp-layout-color2);
}

.jp-CellTags-Applied {
  background-color: var(--jp-layout-color3);
}

.jp-CellTags-Add {
  white-space: nowrap;
  overflow: hidden;
  border: none;
  outline: none;
  resize: horizontal;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color2);
}

.jp-CellTags-Holder {
  display: flex;
  justify-content: center;
  align-items: center;
}

.jp-CellTags-Empty {
  width: 4em;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-completer-item-height: 24px;

  /* Shift the baseline of the type character to align with the match text */
  --jp-private-completer-type-offset: 2px;
}

.jp-Completer {
  box-shadow: var(--jp-elevation-z6);
  background: var(--jp-layout-color1);
  color: var(--jp-content-font-color1);
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding: 0;
  display: flex;
  flex-direction: row;

  /* Needed to avoid scrollbar issues when using cached width. */
  box-sizing: content-box;

  /* Position the completer relative to the text editor, align the '.' */
  margin: 4px 0 0 -30px;
  z-index: 10001;
}

.jp-Completer-docpanel {
  border-left: var(--jp-border-width) solid var(--jp-border-color1);
  width: 400px;
  overflow-y: scroll;
  overflow-x: auto;
  padding: 8px;
  max-height: calc((10 * var(--jp-private-completer-item-height)) - 16px);
}

.jp-Completer-docpanel pre {
  border: none;
  margin: 0;
  padding: 0;
  white-space: pre-wrap;
}

.jp-Completer-list {
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow-y: scroll;
  overflow-x: auto;
  max-height: calc((10 * var(--jp-private-completer-item-height)));
  min-height: calc(var(--jp-private-completer-item-height));
  width: 100%;
}

.jp-Completer-item {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  height: var(--jp-private-completer-item-height);
  min-width: 150px;
  display: grid;
  grid-template-columns: min-content 1fr min-content;
  position: relative;
}

.jp-Completer-item .jp-Completer-match {
  box-sizing: border-box;
  margin: 0;
  padding: 0 8px 0 6px;
  height: var(--jp-private-completer-item-height);
  font-family: var(--jp-code-font-family);
  font-size: var(--jp-code-font-size);
  line-height: var(--jp-private-completer-item-height);
}

.jp-Completer-deprecated .jp-Completer-match {
  text-decoration: line-through;
  color: var(--jp-content-font-color2);
}

.jp-Completer-item .jp-Completer-type {
  box-sizing: border-box;
  height: var(--jp-private-completer-item-height);
  background: transparent;
  width: var(--jp-private-completer-item-height);
}

.jp-Completer-item .jp-Completer-icon {
  /* Normal element size from LabIconStyle.ISheetOptions */
  height: 16px;
  width: 16px;
}

.jp-Completer-item .jp-Completer-monogram {
  text-align: center;
  color: white;
  width: var(--jp-private-completer-item-height);
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  line-height: calc(
    var(--jp-private-completer-item-height) -
      var(--jp-private-completer-type-offset)
  );
  padding-bottom: var(--jp-private-completer-type-offset);
}

.jp-Completer-item .jp-Completer-typeExtended {
  box-sizing: border-box;
  height: var(--jp-private-completer-item-height);
  text-align: right;
  background: transparent;
  color: var(--jp-ui-font-color2);
  font-family: var(--jp-code-font-family);
  font-size: var(--jp-code-font-size);
  line-height: var(--jp-private-completer-item-height);
  padding-right: 8px;
}

.jp-Completer-item:hover {
  background: var(--jp-layout-color2);
  opacity: 0.8;
}

.jp-Completer-item.jp-mod-active {
  background: var(--jp-brand-color1);
  color: white;
}

.jp-Completer-item .jp-Completer-match mark {
  font-weight: bold;
  background: inherit;
  color: inherit;
}

.jp-Completer-type[data-color-index='0'] {
  background: transparent;
}

.jp-Completer-type[data-color-index='1'] {
  background: #1f77b4;
}

.jp-Completer-type[data-color-index='2'] {
  background: #ff7f0e;
}

.jp-Completer-type[data-color-index='3'] {
  background: #2ca02c;
}

.jp-Completer-type[data-color-index='4'] {
  background: #d62728;
}

.jp-Completer-type[data-color-index='5'] {
  background: #9467bd;
}

.jp-Completer-type[data-color-index='6'] {
  background: #8c564b;
}

.jp-Completer-type[data-color-index='7'] {
  background: #e377c2;
}

.jp-Completer-type[data-color-index='8'] {
  background: #7f7f7f;
}

.jp-Completer-type[data-color-index='9'] {
  background: #bcbd22;
}

.jp-Completer-type[data-color-index='10'] {
  background: #17becf;
}

.jp-Completer-loading-bar-container {
  height: 2px;
  width: calc(100% - var(--jp-private-completer-item-height));
  left: var(--jp-private-completer-item-height);
  position: absolute;
  overflow: hidden;
  top: 0;
}

.jp-Completer-loading-bar {
  height: 100%;
  width: 50%;
  background-color: var(--jp-accent-color2);
  position: absolute;
  left: -50%;
  animation: jp-Completer-loading 2s ease-in 0.5s infinite;
}

@keyframes jp-Completer-loading {
  0% {
    transform: translateX(0);
  }

  100% {
    transform: translateX(400%);
  }
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-ConsolePanel {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  margin-top: -1px;
  min-width: 240px;
  min-height: 120px;
}

.jp-CodeConsole {
  height: 100%;
  padding: 0;
  display: flex;
  flex-direction: column;
}

.jp-CodeConsole .jp-Cell {
  padding: var(--jp-cell-padding);
}

/*-----------------------------------------------------------------------------
| Content (already run cells)
|----------------------------------------------------------------------------*/

.jp-CodeConsole-content {
  background: var(--jp-layout-color0);
  flex: 1 1 auto;
  overflow: auto;
  padding: var(--jp-console-padding);
}

.jp-CodeConsole-content .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-inprompt-font-color);
  cursor: move;
}

.jp-CodeConsole-content .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-outprompt-font-color);
}

/* This rule is for styling cell run by another activity in this console */

/* .jp-CodeConsole-content .jp-Cell.jp-CodeConsole-foreignCell {
} */

.jp-CodeConsole-content .jp-InputArea-editor.jp-InputArea-editor {
  background: transparent;
  border: 1px solid transparent;
}

.jp-CodeConsole-content .jp-CodeConsole-banner .jp-InputPrompt {
  display: none;
}

/* collapser is hovered */
.jp-CodeConsole-content .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/*-----------------------------------------------------------------------------
| Input/prompt cell
|----------------------------------------------------------------------------*/

.jp-CodeConsole-input {
  max-height: 80%;
  flex: 0 0 auto;
  overflow: auto;
  border-top: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  padding: var(--jp-cell-padding) var(--jp-console-padding);

  /* This matches the box shadow on the notebook toolbar, eventually we should create
   * CSS variables for this */
  box-shadow: 0 0.4px 6px 0 rgba(0, 0, 0, 0.1);
  background: var(--jp-layout-color1);
}

.jp-CodeConsole-input .jp-CodeConsole-prompt .jp-InputArea {
  height: 100%;
  min-height: 100%;
}

.jp-CodeConsole-promptCell .jp-InputArea-editor.jp-mod-focused {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-CodeConsole {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-CodeConsole .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-CodeConsole .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Private CSS variables */

:root {
  --jp-private-launcher-top-padding: 1.231em;
  --jp-private-launcher-side-padding: 2.462em;
  --jp-private-launcher-card-size: 7.692em;
  --jp-private-launcher-card-label-height: 2.462em;
  --jp-private-launcher-card-icon-height: 5.231em;
  --jp-private-launcher-large-icon-size: 4em;
  --jp-private-launcher-small-icon-size: 2.462em;
}

/* Launcher */

.jp-Launcher {
  margin: 0;
  padding: 0;
  outline: none;
  background: var(--jp-layout-color0);
  box-sizing: border-box;
  min-width: 120px;
  min-height: 120px;

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

.jp-Launcher-body {
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  overflow: auto;
  display: flex;
  justify-content: center;
}

.jp-Launcher-cwd h3 {
  font-size: var(--jp-ui-font-size2);
  font-weight: normal;
  color: var(--jp-ui-font-color1);
  margin: 1em 0;
}

.jp-Launcher-content {
  width: 85%;
  height: 100%;
  padding-top: var(--jp-private-launcher-top-padding);
  padding-left: var(--jp-private-launcher-side-padding);
  padding-right: var(--jp-private-launcher-side-padding);
  box-sizing: border-box;
}

/* Launcher section */

.jp-Launcher-section {
  width: 100%;
  box-sizing: border-box;
  padding-bottom: 12px;
}

.jp-Launcher-sectionHeader {
  display: flex;
  flex-direction: row;
  align-items: center;
  box-sizing: border-box;

  /* This is custom tuned to get the section header to align with the cards */
  margin-left: 5px;
  border-bottom: 1px solid var(--jp-border-color2);
  padding-bottom: 0;
  margin-bottom: 8px;
}

.jp-Launcher-sectionTitle {
  font-size: var(--jp-ui-font-size2);
  font-weight: normal;
  color: var(--jp-ui-font-color0);
  box-sizing: border-box;
}

/* Launcher cards */

.jp-Launcher-cardContainer {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
}

.jp-LauncherCard {
  display: flex;
  flex-direction: column;
  cursor: pointer;
  width: var(--jp-private-launcher-card-size);
  min-height: var(--jp-private-launcher-card-size);
  margin: 8px;
  padding: 0;
  border: 1px solid var(--jp-border-color2);
  background: var(--jp-layout-color0);
  box-shadow: var(--jp-elevation-z2);
  transition: 0.2s box-shadow;
  border-radius: var(--jp-border-radius);
}

.jp-LauncherCard:focus:not(:active) {
  border: 1px solid var(--jp-brand-color1);
  box-shadow: var(--jp-elevation-z6);
  outline: unset; /* if outline is not unset, then depending on which browser you use, the
    border change is either hidden behind the outline or visually combined with it */
}

.jp-LauncherCard:hover {
  box-shadow: var(--jp-elevation-z6);
  background: var(--jp-layout-color1);
}

.jp-LauncherCard:active {
  box-shadow: var(--jp-elevation-z4);
}

.jp-LauncherCard-icon {
  width: 100%;
  height: var(--jp-private-launcher-card-icon-height);
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  display: flex;
  justify-content: center;
  align-items: center;
}

.jp-LauncherCard-noKernelIcon {
  font-weight: normal;
  font-size: var(--jp-private-launcher-large-icon-size);
}

.jp-LauncherCard[data-category='Notebook'] .jp-LauncherCard-noKernelIcon {
  /* This color is copied from the notebook icon. */
  color: var(--jp-jupyter-icon-color);
}

.jp-LauncherCard[data-category='Console'] .jp-LauncherCard-noKernelIcon {
  /* This color is copied from the console icon. */
  color: #0288d1;
}

.jp-LauncherCard-label {
  width: 100%;
  min-height: var(--jp-private-launcher-card-label-height);
  padding: 0 4px 4px;
  box-sizing: border-box;
  overflow: hidden;
}

.jp-LauncherCard-label p {
  min-height: 2.154em;
  word-break: break-word;
  text-align: center;
  color: var(--jp-ui-font-color1);
  line-height: 1.077em;
  font-size: calc(var(--jp-ui-font-size1) * 0.923);
  overflow: hidden;
  margin: auto;
}

/* kernel icons */

.jp-Launcher-kernelIcon {
  width: var(--jp-private-launcher-large-icon-size);
  height: var(--jp-private-launcher-large-icon-size);
  margin: 0;
  padding: 0;

  /* Preserve image aspect ratio */
  object-fit: contain;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-CSVViewer {
  display: flex;
  flex-direction: column;
  outline: none;

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

.jp-CSVDelimiter {
  display: flex;
  flex: 0 0 auto;
  flex-direction: row;
  border: none;
  min-height: 24px;
  background: var(--jp-toolbar-background);
  z-index: 1;
}

.jp-CSVDelimiter .jp-CSVDelimiter-label {
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  padding-left: 8px;
  padding-right: 8px;
}

.jp-CSVDelimiter .jp-CSVDelimiter-dropdown {
  flex: 0 0 auto;
  vertical-align: middle;
  border-radius: 0;
  outline: none;
  height: 20px;
  margin-top: 2px;
  margin-bottom: 2px;
}

.jp-CSVDelimiter .jp-CSVDelimiter-dropdown select.jp-mod-styled {
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size1);
  height: 20px;
  padding-right: 20px;
}

.jp-CSVViewer-grid {
  flex: 1 1 auto;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-FileEditor {
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-FileEditorCodeWrapper {
  overflow: auto;
}

.jp-FileEditorCodeWrapper .cm-editor {
  height: 100%;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-DebuggerBreakpoints {
  display: flex;
  flex-direction: column;
  min-height: 50px;
  padding-top: 3px;
}

.jp-DebuggerBreakpoints-body {
  padding: 10px;
  overflow: auto;
}

.jp-DebuggerBreakpoint {
  display: flex;
  align-items: center;
}

.jp-DebuggerBreakpoint:hover {
  background: var(--jp-layout-color2);
  cursor: pointer;
}

.jp-DebuggerBreakpoint-marker {
  font-size: 20px;
  padding-right: 5px;
  content: '●';
  color: var(--jp-error-color1);
}

.jp-DebuggerBreakpoint-source {
  white-space: nowrap;
  margin-right: 5px;
}

.jp-DebuggerBreakpoint-line {
  margin-left: auto;
}

.jp-DebuggerCallstackFrame {
  display: flex;
  align-items: center;
}

.jp-DebuggerCallstackFrame-name {
  white-space: nowrap;
  margin-right: 5px;
}

.jp-DebuggerCallstackFrame-location {
  margin-left: auto;
}

[data-jp-debugger='true'] .cm-breakpoint-gutter .cm-gutterElement:empty::after {
  content: '●';
  color: var(--jp-error-color1);
  opacity: 0;
}

.cm-gutter {
  cursor: default;
}

.cm-breakpoint-gutter .cm-gutterElement {
  color: var(--jp-error-color1);
  padding-left: 5px;
  font-size: 20px;
  position: relative;
  top: -5px;
}

[data-jp-debugger='true'].jp-Editor
  .cm-breakpoint-gutter
  .cm-gutterElement:empty:hover::after,
[data-jp-debugger='true']
  .jp-Notebook
  .jp-CodeCell.jp-mod-selected
  .cm-breakpoint-gutter:empty:hover::after,
[data-jp-debugger='true']
  .jp-Editor
  .cm-breakpoint-gutter
  .cm-gutterElement:empty:hover::after {
  opacity: 0.5;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-DebuggerCallstack {
  display: flex;
  flex-direction: column;
  min-height: 50px;
  padding-top: 3px;
}

.jp-DebuggerCallstack-body {
  overflow: auto;
}

.jp-DebuggerCallstack-body ul {
  list-style: none;
  margin: 0;
  padding: 0;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
}

.jp-DebuggerCallstack-body li {
  padding: 5px;
  padding-left: 8px;
}

.jp-DebuggerCallstack-body li.selected {
  color: white;
  background: var(--jp-brand-color1);
}

.jp-DebuggerCallstack .jp-ToolbarButtonComponent-label {
  display: none;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-DebuggerEditor-highlight {
  text-shadow: 0 0 1px var(--jp-layout-color0);
  outline: 1px solid;
}

body[data-jp-theme-light='false'] .jp-DebuggerEditor-highlight {
  background-color: var(--md-brown-800);
  outline-color: var(--md-brown-600);
}

body[data-jp-theme-light='true'] .jp-DebuggerEditor-highlight {
  background-color: var(--md-brown-100);
  outline-color: var(--md-brown-300);
}

.jp-DebuggerEditor-marker {
  position: absolute;
  left: -34px;
  top: -1px;
  color: var(--jp-error-color1);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-Button.jp-mod-minimal.jp-TreeView.jp-TreeView,
button.jp-Button.jp-mod-minimal.jp-TreeView.jp-TableView {
  display: flex;
  align-items: center;
  -webkit-transition: 0.4s;
  transition: 0.4s;
  width: 35px;
}

.jp-ViewModeSelected .jp-Button,
.jp-PauseOnExceptions.lm-mod-toggled {
  background-color: var(--jp-inverse-layout-color3);
}

.jp-ViewModeSelected .jp-Button:hover,
.jp-PauseOnExceptions.lm-mod-toggled:hover {
  background-color: var(--jp-inverse-layout-color4);
}

.jp-ViewModeSelected .jp-Button .jp-icon3[fill],
.jp-PauseOnExceptions.lm-mod-toggled .jp-icon3[fill] {
  fill: var(--jp-layout-color1);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-DebuggerKernelSources {
  min-height: 50px;
  margin-top: 3px;
}

[data-jp-debugger='true'].jp-Editor .jp-mod-readOnly {
  background: var(--jp-layout-color2);
  height: 100%;
}

.jp-DebuggerKernelSources-body [data-jp-debugger='true'].jp-Editor {
  height: 100%;
}

.jp-DebuggerKernelSources-body {
  height: 100%;
  overflow-y: auto;
}

.jp-DebuggerKernelSource-filterBox {
  padding: 0;
  flex: 0 0 auto;
  margin: 0;
  position: sticky;
  top: 0;
  background-color: var(--jp-layout-color1);
}

.jp-DebuggerKernelSource-filterBox-hidden {
  display: none;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-SidePanel-header > h2 {
  /* Set font-size to override default h2 sizing but keeping default --jp-ui-font-size0 */
  font-size: 100%;
  font-weight: 600;
  margin: 0 auto 0 0;
  padding: 4px 10px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-DebuggerSources {
  min-height: 50px;
  margin-top: 3px;
}

[data-jp-debugger='true'].jp-Editor .jp-mod-readOnly {
  background: var(--jp-layout-color2);
  height: 100%;
}

.jp-DebuggerSources-body [data-jp-debugger='true'].jp-Editor {
  height: 100%;
}

.jp-DebuggerSources-body {
  height: 100%;
}

.jp-DebuggerSources-header-path {
  overflow: hidden;
  cursor: pointer;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--jp-ui-font-size0);
  color: var(--jp-ui-font-color1);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
.jp-DebuggerVariables {
  display: flex;
  flex-direction: column;
  min-height: 50px;
  padding-top: 3px;
}

.jp-DebuggerVariables-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  min-height: 24px;
  overflow: auto;

  /* For absolute positioning of jp-DebuggerVariables-buttons. */
  position: relative;
}

.jp-DebuggerVariables-branch {
  list-style: none;
  margin: 0;
  padding: 0;
}

.jp-DebuggerVariables-body
  .jp-DebuggerVariables-branch
  .jp-DebuggerVariables-branch {
  grid-area: nested;
}

.jp-DebuggerVariables-body > .jp-DebuggerVariables-branch {
  padding-top: 0.1em;
}

.jp-DebuggerVariables-branch li {
  padding: 3px 0;
  cursor: pointer;
  color: var(--jp-content-font-color1);
  display: grid;
  grid-template:
    'collapser name detail'
    'nested nested nested';
  grid-template-columns: max-content max-content 1fr;
}

.jp-DebuggerVariables-branch li:not(:has(li:hover)):hover {
  background: var(--jp-layout-color2);
}

.jp-DebuggerVariables-collapser {
  width: 16px;
  grid-area: collapser;
  transform: rotate(-90deg);
  transition: transform 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

.jp-DebuggerVariables-collapser.jp-mod-expanded {
  transform: rotate(0);
}

.jp-DebuggerVariables-buttons {
  position: absolute;
  top: 0;
  right: 8px;
  margin-top: 1px;
}

.jp-DebuggerVariables-name {
  color: var(--jp-mirror-editor-attribute-color);
  grid-area: name;
}

.jp-DebuggerVariables-name::after {
  content: ':';
  margin-right: 5px;
}

.jp-DebuggerVariables-detail {
  /* detail contains value for primitive types or name of the type otherwise */
  color: var(--jp-mirror-editor-string-color);
}

.jp-DebuggerVariables-renderVariable {
  border: none;
  background: none;
  cursor: pointer;
  transform-origin: center center;
  transition: transform 0.2s cubic-bezier(0.4, 0, 1, 1);
}

.jp-DebuggerVariables-renderVariable:active {
  transform: scale(1.35);
}

.jp-DebuggerVariables-renderVariable:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DebuggerVariables-renderVariable:disabled {
  cursor: default;
}

.jp-DebuggerVariables-branch li > .jp-DebuggerVariables-branch {
  margin-left: 12px;
}

.jp-DebuggerVariables-grid {
  flex: 1 1 auto;
}

.jp-DebuggerVariables-grid .lm-DataGrid {
  border: none;
}

.jp-DebuggerVariables-colorPalette {
  visibility: hidden;
  z-index: -999;
  position: absolute;
  left: -999px;
  top: -999px;
}

.jp-DebuggerVariables-colorPalette .jp-mod-void {
  color: var(--jp-layout-color1);
}

.jp-DebuggerVariables-colorPalette .jp-mod-background {
  color: var(--jp-rendermime-table-row-background);
}

.jp-DebuggerVariables-colorPalette .jp-mod-header-background {
  color: var(--jp-layout-color2);
}

.jp-DebuggerVariables-colorPalette .jp-mod-grid-line {
  color: var(--jp-border-color3);
}

.jp-DebuggerVariables-colorPalette .jp-mod-header-grid-line {
  color: var(--jp-border-color3);
}

.jp-DebuggerVariables-colorPalette .jp-mod-selection {
  /* TODO: Fix JupyterLab light theme (alpha) so this can be a variable. */
  color: rgba(3, 169, 244, 0.2);
}

.jp-DebuggerVariables-colorPalette .jp-mod-text {
  color: var(--jp-content-font-color0);
}

.jp-VariableRendererPanel {
  overflow: auto;
}

.jp-VariableRendererPanel-renderer {
  overflow: auto;
  height: 100%;
}

.jp-VariableRenderer-TrustButton[aria-pressed='true'] {
  box-shadow: inset 0 var(--jp-border-width) 4px
    rgba(
      var(--jp-shadow-base-lightness),
      var(--jp-shadow-base-lightness),
      var(--jp-shadow-base-lightness),
      0.6
    );
}

.jp-DebuggerRichVariable div[data-mime-type='text/plain'] > pre {
  white-space: normal;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-left-truncated {
  overflow: hidden;
  text-overflow: ellipsis;
  direction: rtl;
}

#jp-debugger .jp-switch-label {
  margin-right: 0;
}

.jp-DebuggerBugButton[aria-pressed='true'] path {
  fill: var(--jp-warn-color0);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-LogConsolePanel {
  overflow-y: auto;
}

.jp-LogConsolePanel .jp-OutputArea-child {
  border-bottom: 1px solid var(--jp-border-color3);
}

.jp-LogConsolePanel .jp-OutputArea-prompt {
  width: 85px;
  color: var(--jp-ui-font-color2);
  font-size: 13px;
  padding: 2px;
}

.jp-LogConsolePanel .jp-OutputArea-prompt[data-log-level='info'] {
  background-color: var(--jp-info-color1);
  color: var(--jp-ui-inverse-font-color1);
}

.jp-LogConsolePanel .jp-OutputArea-prompt[data-log-level='warning'] {
  background-color: var(--jp-warn-color1);
  color: var(--jp-ui-inverse-font-color1);
}

.jp-LogConsolePanel .jp-OutputArea-prompt[data-log-level='error'] {
  background-color: var(--jp-error-color1);
  color: var(--jp-ui-inverse-font-color1);
}

.jp-LogConsolePanel .jp-OutputArea-prompt[data-log-level='critical'] {
  background-color: var(--jp-error-color0);
  color: var(--jp-ui-inverse-font-color0);
}

.jp-LogConsoleListPlaceholder {
  padding: 5px;
  font-size: 13px;
  color: var(--jp-ui-font-color3);
}

.jp-Scrolling {
  overflow-y: auto;
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
  Main widget layout and styling
*/

.jp-extensionmanager-view .lm-SplitPanel-child {
  overflow-y: auto;
}

.jp-extensionmanager-header {
  width: 100%;
}

.jp-extensionmanager-title {
  display: flex;
  margin: 4px 10px 0;
}

.jp-extensionmanager-title svg {
  width: 14px;
}

.jp-extensionmanager-title .jp-icon3[fill] {
  fill: currentcolor;
}

.jp-extensionmanager-path {
  display: inline flex;
  padding-left: 4px;
}

.jp-extensionmanager-header .jp-FilterBox {
  margin: 4px 10px 0;
}

.jp-extensionmanager-disclaimer {
  padding: 0 8px;
  min-height: 160px;
}

.jp-extensionmanager-disclaimer a {
  color: var(--jp-content-link-color);
}

.jp-extensionmanager-disclaimer a:hover {
  color: var(--jp-content-link-color);
  text-decoration: underline;
}

.jp-extensionmanager-disclaimer .jp-Button {
  margin: 0 5px;
}

/*
  List view layout and styling
*/

.jp-extensionmanager-listview-wrapper {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  height: 100%;
}

.jp-extensionmanager-listview {
  list-style-type: none;
  margin: 0;
  padding: 0;
  min-height: 0;
  flex: 1 1 auto;
  overflow: auto;
}

/*
  Section headers
*/

.jp-extensionmanager-view .jp-SidePanel-header .jp-extensionmanager-error {
  text-transform: none;
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
}

/*
  Error messages
*/

.jp-extensionmanager-view .jp-extensionmanager-error pre {
  white-space: pre-wrap;
}

/*
  List view pagination styling
*/

.jp-extensionmanager-view ul.pagination {
  display: flex;
  justify-content: center;
  padding-left: 0;
  padding-right: 0;
}

.jp-extensionmanager-pagination li {
  display: inline-block;
}

/* stylelint-disable selector-max-type */
.jp-extensionmanager-pagination li > a {
  padding: 0 5px;
  cursor: pointer;
}
/* stylelint-enable selector-max-type */

.jp-extensionmanager-pagination li.active > a {
  background: var(--jp-brand-color1);
  color: var(--jp-layout-color1);
}

.jp-extensionmanager-pagination li > a:hover {
  background-color: var(--jp-layout-color2);
}

.jp-extensionmanager-pagination .break > a {
  cursor: default;
}

.jp-extensionmanager-installedlist,
.jp-extensionmanager-searchresults {
  min-height: 100px;
}

/*
  Entry layout and styling
*/

.jp-extensionmanager-entry {
  padding: 8px;
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
}

.jp-extensionmanager-entry-description {
  min-width: 0;
  flex-grow: 1;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color2);
  font-weight: 400;
}

.jp-extensionmanager-entry-title {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
}

.jp-extensionmanager-entry.jp-extensionmanager-entry-should-be-uninstalled {
  background-color: var(--jp-error-color3);
}

/* Precedence order update/error/warning matters! */

.jp-extensionmanager-entry.jp-extensionmanager-entry-update {
  border-left: solid 8px var(--jp-brand-color2);
  padding-left: 4px;
}

.jp-extensionmanager-entry.jp-extensionmanager-entry-error {
  border-left: solid 8px var(--jp-error-color2);
  padding-left: 4px;
}

.jp-extensionmanager-entry-name {
  font-size: var(--jp-ui-font-size1);
  font-weight: 600;
  padding: 0 8px 0 0;
  margin-bottom: 4px;
  overflow-wrap: anywhere;
}

.jp-extensionmanager-entry-name a:link {
  color: var(--jp-content-link-color);
  text-decoration: none;
}

.jp-extensionmanager-entry-name a:visited {
  color: var(--jp-content-link-color);
  text-decoration: none;
}

.jp-extensionmanager-entry-name a:hover {
  color: var(--jp-content-link-color);
  text-decoration: underline;
}

.jp-extensionmanager-loader,
.jp-extensionmanager-error,
.jp-extensionmanager-listview-message {
  padding: 4px 12px;
}

.jp-extensionmanager-listview-message {
  font-size: var(--jp-ui-font-size1);
}

.jp-extensionmanager-entry-content {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

/*
  Pending entry progress indicator
*/

.jp-extensionmanager-pending {
  height: 4px;
  width: 100%;
  position: relative;
  overflow: hidden;
  flex: 0 0 auto;
  background-color: var(--jp-layout-color1);
}

.jp-extensionmanager-pending.jp-mod-hasPending::before {
  display: block;
  position: absolute;
  content: '';
  left: -200px;
  width: 200px;
  height: 4px;
  border-radius: 4px;
  background-color: var(--jp-brand-color1);
  animation: loading 2s linear infinite;
}

@keyframes loading {
  0% {
    left: -200px;
    width: 30%;
  }

  50% {
    width: 30%;
  }

  70% {
    width: 70%;
  }

  80% {
    left: 50%;
  }

  95% {
    left: 120%;
  }

  100% {
    left: 100%;
  }
}

/*
  Disclaimer buttons
*/

.jp-extensionmanager-disclaimer-disable {
  background-color: var(--jp-reject-color-normal) !important;
  color: white !important;
  border: 0;
  background-image: none !important;
}

.jp-extensionmanager-disclaimer-enable {
  background-color: var(--jp-error-color1) !important;
  color: white !important;
  border: 0;
  background-image: none !important;
}

/*
  Entry buttons layout and styling
*/

.jp-extensionmanager-entry-buttons {
  display: flex;
  flex-direction: row;
  padding-block-start: 4px;
  justify-content: flex-end;
}

/*
  Rebuild prompt dialog
*/

.jp-extensionmanager-buildprompt {
  background-color: var(--jp-brand-color1);
  color: var(--jp-ui-inverse-font-color1);
  padding: 4px 8px;
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
}

.jp-extensionmanager-buildprompt button {
  border: none;
  background-color: transparent;
  color: var(--jp-ui-inverse-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 600;
  float: right;
  margin: 4px;
}

/*
  Generic dialog
*/

.jp-extensionmanager-dialog .jp-extensionmanager-dialog-subheader {
  font-size: var(--jp-ui-font-size1);
  font-weight: 600;
  color: var(--jp-ui-font-color0);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

#filebrowser.jp-mod-restoring .jp-DirListing-content {
  display: none;
}

.jp-FileBrowser-filterBox {
  padding: 0;
  flex: 0 0 auto;
}

.jp-FileBrowser-filterBox input {
  line-height: 24px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Document styles */
.jp-HTMLViewer {
  overflow: hidden;
  min-width: 100px;
  min-height: 100px;
  width: 100%;
  height: 100%;
}

.jp-HTMLViewer > iframe {
  border: none;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-HTMLViewer {
  position: relative;
}

body.lm-mod-override-cursor .jp-HTMLViewer::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-ImageViewer {
  overflow: auto;
}

.jp-ImageViewer > img {
  box-sizing: border-box;
  transform-origin: top left;
}
</style><style>/* Copyright (c) Jupyter Development Team.
Distributed under the terms of the Modified BSD License. */

.jp-RenderedJavaScript {
  padding: var(--jp-code-padding);
}
</style><style>/**
  Copyright (c) Jupyter Development Team.
  Distributed under the terms of the Modified BSD License.
*/

/* Base styles */
.jp-RenderedJSON {
  width: 100%;
  height: 100%;
  padding: 0;
  overflow: hidden;
}

/* Override react-json-tree inline styles */
.jp-RenderedJSON *:not(mark) {
  background-color: transparent !important;
}

.jp-RenderedJSON ul {
  margin: 0 !important;
}

.jp-RenderedJSON .container {
  position: relative;
  width: 100%;
  min-height: 30px;
}

.jp-RenderedJSON .filter {
  position: absolute;
  top: 0;
  right: 0;
  width: 33%;
  max-width: 150px;
  z-index: 10;
}

/* Document styles */
.jp-MimeDocument .jp-RenderedJSON {
  padding: 5px 5px 5px 20px;
  overflow-y: auto;
}

.jp-RenderedJSON mark.jp-mod-selected {
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

/* Output styles */

/* .jp-OutputArea .jp-RenderedJSON {

} */
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-running-item-height: 24px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-RunningSessions {
  display: flex;
  flex-direction: column;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

.jp-RunningSessions > .jp-SidePanel-toolbar {
  justify-content: flex-end;
}

.jp-RunningSessions-section {
  min-height: 50px;
  overflow: auto;
}

.jp-RunningSessions-sectionContainer {
  margin: 0;
  padding: 0;
  overflow: auto;
}

.jp-RunningSessions-sectionList {
  display: block;
  margin: 0;
  padding: 0;
  list-style-type: none;
}

.jp-RunningSessions-item {
  display: flex;
  flex-direction: row;
  color: var(--jp-ui-font-color1);
  height: var(--jp-private-running-item-height);
  line-height: var(--jp-private-running-item-height);
  padding: 0 8px;
}

.jp-RunningSessions-item:hover {
  background-color: var(--jp-layout-color2);
  cursor: pointer;
}

.jp-RunningSessions-item.jp-mod-running-child {
  padding-left: 40px;
}

.jp-RunningSessions-item.jp-mod-kernel:not(.jp-mod-running-child) {
  user-select: none;
}

.jp-RunningSessions-item img {
  height: 16px;
  margin-top: 4px;
}

.jp-RunningSessions-itemLabel {
  font-size: var(--jp-ui-font-size1);
  flex: 1 1 55%;
  padding: 0 4px;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

.jp-RunningSessions-itemLabel:focus {
  background-color: var(--jp-layout-color2);
}

.jp-RunningSessions-itemDetail {
  font-size: var(--jp-ui-font-size1);
  flex: 1 1 45%;
  padding: 0 4px;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

.jp-RunningSessions-item .jp-RunningSessions-itemShutdown {
  border-radius: 0;
}

.jp-RunningSessions-item:not(:hover) .jp-RunningSessions-itemShutdown {
  visibility: hidden;
}

.jp-RunningSessions-sectionList
  .jp-RunningSessions-item
  .jp-Button.jp-RunningSessions-itemShutdown:hover {
  background: var(--jp-layout-color3);
}

.jp-RunningSessions-shutdownAll.jp-mod-styled
  > .jp-ToolbarButtonComponent-label {
  color: var(--jp-warn-color1);
  background-color: transparent;
  border-radius: 2px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-RunningSessions-shutdownAll.jp-mod-styled:hover
  > .jp-ToolbarButtonComponent-label {
  background-color: var(--jp-layout-color2);
}

.jp-RunningSessions-shutdownAll.jp-mod-styled:focus
  > .jp-ToolbarButtonComponent-label {
  border: none;
  box-shadow: none;
  background-color: var(--jp-layout-color2);
}

.jp-RunningSessions-shutdownAll.jp-mod-styled.jp-mod-disabled
  > .jp-ToolbarButtonComponent-label {
  color: var(--jp-ui-font-color2);
}

.jp-RunningSessions-shutdownAll.jp-mod-styled.jp-mod-disabled:hover
  > .jp-ToolbarButtonComponent-label {
  background: none;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-LSPExtension-FormGroup-content > select {
  margin: 0 12px;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-markdownviewer-padding: 32px;
}

.jp-Document .jp-MarkdownViewer .jp-RenderedMarkdown {
  padding-top: var(--jp-private-markdownviewer-padding);
  padding-right: var(--jp-private-markdownviewer-padding);
  padding-bottom: var(--jp-private-markdownviewer-padding);
  padding-left: var(--jp-private-markdownviewer-padding);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-MarkdownViewer .jp-RenderedHTMLCommon {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}
</style><style>/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

@font-face /* 0 */ {
  font-display: swap;
  font-family: MJXZERO;
  src: url(http://localhost:8888/static/notebook/481e39042508ae313a60.woff)
    format('woff');
}

@font-face /* 1 */ {
  font-display: swap;
  font-family: MJXTEX;
  src: url(http://localhost:8888/static/notebook/1cb1c39ea642f26a4dfe.woff)
    format('woff');
}

@font-face /* 2 */ {
  font-display: swap;
  font-family: MJXTEX-B;
  src: url(http://localhost:8888/static/notebook/88b98cad3688915e50da.woff)
    format('woff');
}

@font-face /* 3 */ {
  font-display: swap;
  font-family: MJXTEX-I;
  src: url(http://localhost:8888/static/notebook/a009bea404f7a500ded4.woff)
    format('woff');
}

@font-face /* 4 */ {
  font-display: swap;
  font-family: MJXTEX-MI;
  src: url(http://localhost:8888/static/notebook/355254db9ca10a09a3b5.woff)
    format('woff');
}

@font-face /* 5 */ {
  font-display: swap;
  font-family: MJXTEX-BI;
  src: url(http://localhost:8888/static/notebook/8ea8dbb1b02e6f730f55.woff)
    format('woff');
}

@font-face /* 6 */ {
  font-display: swap;
  font-family: 'MJXTEX-S1';
  src: url(http://localhost:8888/static/notebook/c49810b53ecc0d87d802.woff)
    format('woff');
}

@font-face /* 7 */ {
  font-display: swap;
  font-family: 'MJXTEX-S2';
  src: url(http://localhost:8888/static/notebook/30e889b58cbc51adfbb0.woff)
    format('woff');
}

@font-face /* 8 */ {
  font-display: swap;
  font-family: 'MJXTEX-S3';
  src: url(http://localhost:8888/static/notebook/5cda41563a095bd70c78.woff)
    format('woff');
}

@font-face /* 9 */ {
  font-display: swap;
  font-family: 'MJXTEX-S4';
  src: url(http://localhost:8888/static/notebook/3bc6ecaae7ecf6f8d7f8.woff)
    format('woff');
}

@font-face /* 10 */ {
  font-display: swap;
  font-family: MJXTEX-A;
  src: url(http://localhost:8888/static/notebook/3de784d07b9fa8f104c1.woff)
    format('woff');
}

@font-face /* 11 */ {
  font-display: swap;
  font-family: MJXTEX-C;
  src: url(http://localhost:8888/static/notebook/26683bf201fb258a2237.woff)
    format('woff');
}

@font-face /* 12 */ {
  font-display: swap;
  font-family: MJXTEX-CB;
  src: url(http://localhost:8888/static/notebook/af04542b29eaac04550a.woff)
    format('woff');
}

@font-face /* 13 */ {
  font-display: swap;
  font-family: MJXTEX-FR;
  src: url(http://localhost:8888/static/notebook/870673df72e70f87c91a.woff)
    format('woff');
}

@font-face /* 14 */ {
  font-display: swap;
  font-family: MJXTEX-FRB;
  src: url(http://localhost:8888/static/notebook/721921bab0d001ebff02.woff)
    format('woff');
}

@font-face /* 15 */ {
  font-display: swap;
  font-family: MJXTEX-SS;
  src: url(http://localhost:8888/static/notebook/b418136e3b384baaadec.woff)
    format('woff');
}

@font-face /* 16 */ {
  font-display: swap;
  font-family: MJXTEX-SSB;
  src: url(http://localhost:8888/static/notebook/32792104b5ef69eded90.woff)
    format('woff');
}

@font-face /* 17 */ {
  font-family: MJXTEX-SSI;
  src: url(http://localhost:8888/static/notebook/fc6ddf5df402b263cfb1.woff)
    format('woff');
}

@font-face /* 18 */ {
  font-display: swap;
  font-family: MJXTEX-SC;
  src: url(http://localhost:8888/static/notebook/af96f67d7accf5fd2a4a.woff)
    format('woff');
}

@font-face /* 19 */ {
  font-display: swap;
  font-family: MJXTEX-T;
  src: url(http://localhost:8888/static/notebook/c56da8d69f1a0208b8e0.woff)
    format('woff');
}

@font-face /* 20 */ {
  font-display: swap;
  font-family: MJXTEX-V;
  src: url(http://localhost:8888/static/notebook/72bc573386dd1d48c5bb.woff)
    format('woff');
}

@font-face /* 21 */ {
  font-display: swap;
  font-family: MJXTEX-VB;
  src: url(http://localhost:8888/static/notebook/36e0d72d8a7afc696a3e.woff)
    format('woff');
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-RenderedLatex .MJX-DISPLAY {
  margin: 0;
  text-align: left;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2019, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MetadataForm {
  width: 100%;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
}

.jp-MetadataForm-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
}

.jp-MetadataForm-placeholderContent {
  padding: 8px;
}

.jp-MetadataForm .panel-danger.errors {
  display: none;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-PDFContainer iframe {
  position: absolute;
  z-index: 0;
  top: 0;
  right: 0;
  width: 100%;
  height: 100%;
}

.jp-OutputArea .jp-PDFContainer {
  min-height: 512px;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
This reuses the same CSS selector logic as jp-IFrame to prevent embedded
PDFs from swallowing cursor events.
*/
body.lm-mod-override-cursor .jp-PDFContainer {
  position: relative;
}

body.lm-mod-override-cursor .jp-PDFContainer::before {
  content: '';
  position: absolute;
  z-index: 10;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-inspector-tab-height: 24px;
  --jp-private-inspector-tab-width: 60px;
}

/*-----------------------------------------------------------------------------
| Inspector
|----------------------------------------------------------------------------*/

.jp-Inspector {
  outline: none;
  min-width: 120px;
  min-height: 120px;
}

.jp-Inspector-content {
  background: var(--jp-layout-color1);
  border: none;
  padding: 8px;
  width: 100%;
  height: 100%;
  overflow: auto;
}

.jp-Inspector-content pre {
  font-size: var(--jp-code-font-size);
  line-height: var(--jp-code-line-height);
  margin: 0;
  white-space: pre-wrap;
}

.jp-Inspector-default-content {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: xx-large;
  font-style: italic;
  color: darkgray;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-settingeditor-debug-height: 28px;
  --jp-private-settingeditor-key-width: 150px;
  --jp-private-settingeditor-legend-height: 16px;
  --jp-private-settingeditor-row-height: 16px;
  --jp-private-settingeditor-toolbar-height: 28px;
  --jp-private-settingeditor-type-width: 75px;
  --jp-private-settingeditor-modifier-indent: 5px;
  --jp-private-settingeditor-header-spacing: 8px;
}

.jp-SettingsPanel,
#json-setting-editor {
  min-width: 360px;
  min-height: 240px;
  background-color: var(--jp-layout-color0);
  color: var(--jp-ui-font-color0);
  margin-top: -1px;
  outline: none;

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

#setting-editor > .lm-Widget,
#json-setting-editor > .lm-Widget {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  right: 0;
}

#setting-editor .lm-SplitPanel-handle,
#json-setting-editor .lm-SplitPanel-handle {
  background-color: var(--jp-border-color2);
}

/** Plugin list **/

.jp-PluginList {
  min-width: 175px;
  max-width: 275px;
  background-color: var(--jp-layout-color2);
}

.jp-PluginList-wrapper {
  overflow-y: auto;
  height: 100%;
}

.jp-PluginList ul {
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  list-style-type: none;
  margin: 0;
  padding: 0;
  overflow-y: auto;
  position: relative;
  top: 0;
  bottom: 0;
  left: 0;
  right: 0;
}

.jp-PluginList .jp-PluginList-header {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color2);
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
}

.jp-PluginList .jp-PluginList-noResults,
.jp-PluginList .jp-PluginList-header {
  flex: 0 0 auto;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-size: var(--jp-ui-font-size0);
  padding: 8px 8px 8px 12px;
  margin: 10px;
  border-bottom: var(--jp-border-width) solid var(--jp-border-color2);
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
}

.jp-PluginList .jp-SelectedIndicator {
  width: 3px;
  background-color: var(--jp-brand-color1);
  height: var(--jp-cell-collapser-min-height);
  visibility: hidden;
}

.jp-PluginList .jp-mod-selected .jp-SelectedIndicator {
  visibility: inherit;
}

.jp-PluginList .jp-ErrorPlugin .jp-SelectedIndicator {
  background-color: var(--jp-error-color0);
}

.jp-PluginList-icon {
  display: flex;
  height: 20px;
  width: 20px;
  margin-right: 3px;
  position: relative;
}

.jp-PluginList-wrapper > .jp-FilterBox {
  margin: 8px 12px 0;
}

.jp-PluginList mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.jp-PluginList-entry {
  display: flex;
  flex-direction: column;
  border: 1px solid transparent;
  background: transparent;
  overflow: hidden;
  padding: 4px 0 4px 4px;
  white-space: nowrap;
}

.jp-PluginList-entry:hover {
  background: var(--jp-layout-color1);
}

.jp-PluginList-entry li {
  margin-left: 27px;
  margin-top: 5px;
  color: var(--jp-ui-font-color1);
  overflow-x: hidden;
  text-overflow: ellipsis;
}

.jp-PluginList-entry-label {
  display: flex;
}

.jp-PluginList-entry-label-text {
  text-overflow: ellipsis;
  overflow-x: hidden;
  white-space: nowrap;
  color: var(--jp-ui-font-color1);
  line-height: var(--jp-cell-collapser-min-height);
}

/** Raw editor **/

.jp-SettingsRawEditor .jp-Toolbar {
  color: var(--jp-ui-font-color0);
  font-size: var(--jp-ui-font-size1);
  height: var(--jp-private-settingeditor-toolbar-height);
  max-height: var(--jp-private-settingeditor-toolbar-height);
}

.jp-SettingsRawEditor .jp-Toolbar .jp-ToolbarButtonComponent-label {
  display: none;
}

.jp-SettingsRawEditor .jp-Toolbar-item {
  margin-top: 1px;
  align-items: center;
}

.jp-SettingsRawEditor .jp-Inspector {
  border-top: 2px solid var(--jp-layout-color2);
  min-height: var(--jp-private-settingeditor-debug-height);
  max-height: var(--jp-private-settingeditor-debug-height);
}

.jp-SettingsRawEditor .jp-Inspector.jp-SettingsDebug .jp-RenderedHTMLCommon {
  padding: 2px 5px 2px 0;
  width: 100%;
}

.jp-SettingsRawEditor .jp-Inspector.jp-SettingsDebug .jp-RenderedHTMLCommon p {
  text-align: right;
}

.jp-SettingsRawEditor .cm-editor {
  height: 100%;
}

/** Panel **/

.jp-SettingsPanel .checkbox p {
  font-size: var(--jp-content-font-size1);
}

.jp-SettingsPanel .checkbox {
  display: flex;
  flex-direction: column-reverse;
}

.jp-SettingsPanel .form-group {
  display: flex;
  padding: 4px 8px 4px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 5px;
}

.jp-SettingsPanel .jp-SettingsEditor {
  padding: 20px;
}

.jp-SettingsPanel {
  overflow-y: auto;
  height: 100%;
}

.jp-SettingsForm {
  position: relative;
}

.jp-SettingsForm > .rjsf > .form-group {
  padding-top: 0;
  margin-top: 0;
}

/** Settings header **/

.jp-SettingsHeader {
  display: grid;
  grid-template:
    'title buttonbar'
    'description buttonbar';
  grid-template-columns: 1fr max-content;
  padding: 0 var(--jp-private-settingeditor-header-spacing);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-SettingsHeader-title {
  font-size: var(--jp-content-font-size3);
  color: var(--jp-ui-font-color0);
  font-weight: 400;
  grid-area: title;
  padding: 0;
  margin-top: calc(var(--jp-private-settingeditor-header-spacing) * 2);
  margin-bottom: calc(var(--jp-private-settingeditor-header-spacing) / 2);
}

.jp-SettingsHeader-description {
  grid-area: description;
  padding-bottom: var(--jp-private-settingeditor-header-spacing);
  color: var(--jp-ui-font-color1);
}

.jp-SettingsHeader-buttonbar {
  margin: auto var(--jp-private-settingeditor-header-spacing);
  grid-row: span 2;
}

.jp-SettingsHeader-buttonbar > .jp-RestoreButton {
  background-color: var(--jp-warn-color-normal);
  border: 0;
  color: var(--jp-ui-inverse-font-color0);
}

.jp-PluginEditor {
  overflow: auto;
}

/** Placeholder **/

.jp-SettingsEditor-placeholder {
  text-align: center;
}

.jp-SettingsEditor-placeholderContent {
  color: var(--jp-content-font-color2);
  padding: 8px;
}

.jp-SettingsEditor-placeholderContent > h3 {
  margin-bottom: var(--jp-content-heading-margin-bottom);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Shortcut Input Style */

.jp-Shortcuts-InputBox {
  display: inline-flex;
  padding-top: 2px;
}

.jp-Shortcuts-InputBoxNew {
  margin-left: 10px;
}

.jp-mod-hidden {
  display: none;
}

@keyframes slide-animation {
  from {
    width: 0;
    left: 0;
  }

  to {
    width: 120px;
    left: 0;
  }
}

.jp-Shortcuts-Input {
  animation-duration: 0.5s;
  animation-timing-function: ease-out;
  animation-name: slide-animation;
  border-width: var(--jp-border-width);
  border-color: var(--jp-border-color3);
  border-style: solid;
  background-color: var(--jp-layout-color0);
  margin-left: auto;
  padding-left: 10px;
  width: 120px;
  height: 25px;
  line-height: 25px;
  display: block;
}

.jp-Shortcuts-Input:focus {
  outline: none;
  color: var(--jp-content-font-color1);
  border-color: var(--jp-brand-color2);
}

.jp-mod-unavailable-Input:focus {
  border-color: var(--jp-error-color2);
}

.jp-Shortcuts-InputText {
  overflow-x: hidden;
  overflow-y: hidden;
  margin: 0;
  margin-top: 4px;
  padding: 0 5px;
  height: 17px;
  line-height: 17px;
  width: fit-content;
}

.jp-mod-selected-InputText {
  background-color: var(--jp-brand-color3);
  overflow: hidden;
}

.jp-mod-waiting-InputText {
  color: var(--jp-content-font-color3);
}

.jp-Shortcuts-Submit {
  background-color: var(--jp-brand-color2);
  border-radius: 0;
  border: none;
  color: var(--jp-layout-color0);
  font-family: var(--jp-ui-font-family);
  display: block;
  height: 27px;
  width: 26px;
  cursor: pointer;
}

.jp-Shortcuts-Submit:focus {
  outline: none;
}

.jp-Shortcuts-Submit .jp-icon3[fill] {
  fill: var(--jp-layout-color1);
}

.jp-Shortcuts-Submit.jp-mod-defunc-Submit {
  background-color: var(--jp-layout-color3);
}

.jp-Shortcuts-Submit.jp-mod-defunc-Submit .jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-Shortcuts-Submit.jp-mod-conflict-Submit {
  background-color: var(--jp-error-color1);
}

/* Shortcut Item Style */
.jp-Shortcuts-Cell {
  padding: 6px 12px;
  display: table-cell;
  width: 20%;
  vertical-align: middle;
}

.jp-Shortcuts-ShortcutCell {
  display: flex;
  min-width: 100px;
  flex-wrap: wrap;
}

.jp-Shortcuts-EmptyCell {
  height: 32px;
}

.jp-Shortcuts-Row {
  padding: 10px;
  width: 100%;
  display: table-row;
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: middle;
  background-color: var(--jp-layout-color0);
}

.jp-Shortcuts-Row:hover .jp-Shortcuts-ShortcutKeys {
  border-color: var(--jp-border-color1);
  background: var(--jp-layout-color2);
}

.jp-Shortcuts-Row:hover #add-link,
.jp-Shortcuts-Row:hover #or {
  display: block;
}

.jp-Shortcuts-ErrorMessage {
  color: var(--jp-error-color1);
  margin-top: 9px;
}

.jp-Shortcuts-ErrorButton {
  line-height: 34px;
  margin-left: 10px;
}

.jp-Shortcuts-ErrorButton button:nth-of-type(1) {
  height: 25px;
  margin-right: 5px;
  background-color: var(--jp-border-color0);
  color: white;
  outline: none;
}

.jp-Shortcuts-ErrorButton button:nth-of-type(1):active,
.jp-Shortcuts-ErrorButton button:nth-of-type(1):focus {
  outline: none;
  border: none;
}

.jp-Shortcuts-ErrorButton button:nth-of-type(2) {
  height: 25px;
  background-color: var(--jp-error-color1);
  color: white;
  outline: none;
}

.jp-Shortcuts-ErrorButton button:nth-of-type(2):active,
.jp-Shortcuts-ErrorButton button:nth-of-type(2):focus {
  outline: none;
  border: none;
}

.jp-Shortcuts-ShortcutContainer {
  display: flex;
  flex-wrap: wrap;
}

.jp-Shortcuts-ShortcutContainer:hover .jp-Shortcuts-ShortcutKeys {
  border-color: var(--jp-border-color3);
  background: var(--jp-layout-color3);
}

.jp-Shortcuts-ShortcutKeysContainer {
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-ui-font-family);
  display: flex;
}

.jp-Shortcuts-ConflictContainer {
  display: flex;
  flex-wrap: wrap;
  padding: 6px 12px;
  margin-left: 20%;
}

.jp-Shortcuts-ShortcutKeys {
  border-width: var(--jp-border-width);
  border-color: var(--jp-layout-color0);
  border-radius: var(--jp-border-radius);
  padding: 5px 6px;
  margin: 3px 0;
}

.jp-Shortcuts-Or {
  margin-right: 12px;
  margin-left: 12px;
  margin-top: 8px;
  color: var(--jp-content-font-color3);
  display: none;
}

.jp-Shortcuts-Or:hover {
  display: block;
}

.jp-Shortcuts-OrTwo {
  margin-right: 12px;
  margin-left: 12px;
  margin-top: 8px;
  color: var(--jp-content-font-color3);
  display: block;
}

.jp-Shortcuts-Comma {
  margin-top: 10px;
  margin-right: 2px;
  margin-left: 2px;
}

.jp-Shortcuts-Plus {
  display: none;
  background: var(--jp-brand-color3);
  border-color: var(--jp-layout-color0);
  border-radius: var(--jp-border-radius);
  border-width: var(--jp-border-width);
  margin: 3px 0;
  padding: 5px 6px;
}

.jp-Shortcuts-Plus:hover {
  background-color: var(--jp-brand-color2);
}

.jp-Shortcuts-Plus:active {
  background-color: var(--jp-brand-color2);
}

.jp-Shortcuts-Reset {
  color: var(--jp-brand-color2);
  padding-left: 10px;
}

.jp-Shortcuts-Reset:hover {
  color: var(--jp-brand-color1);
}

.jp-Shortcuts-SourceCell {
  display: inline-block;
}

/* Shortcut List Style */
.jp-Shortcuts-ShortcutList {
  width: 100%;
  display: table;
  border-collapse: collapse;
}

.jp-Shortcuts-ShortcutListContainer {
  overflow-y: scroll;
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

/* Shortcut Title Item Style */
.jp-Shortcuts-Header {
  display: flex;
  cursor: pointer;
}

.jp-Shortcuts-Header:hover .jp-ShortcutTitleItem-sortButton .jp-icon3[fill],
.jp-Shortcuts-Header:focus .jp-ShortcutTitleItem-sortButton .jp-icon3[fill] {
  fill: var(--jp-ui-font-color0);
}

.jp-Shortcuts-Header:active .jp-ShortcutTitleItem-sortButton {
  outline: none;
}

.jp-Shortcuts-CurrentHeader .jp-icon3[fill] {
  fill: var(--jp-ui-font-color0);
}

/* Shortcut UI Style */

.jp-Shortcuts-ShortcutUI {
  display: flex;
  flex-direction: column;
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  color: var(--jp-content-font-color1);
  min-width: 450px;
  width: 100%;
}

/* TopNav Style */
.jp-Shortcuts-Top {
  display: block;
}

.jp-Shortcuts-TopNav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-sizing: border-box;
  font-size: var(--jp-ui-font-size1);
  background-color: var(--jp-layout-color0);
}

.jp-Shortcuts-Symbols {
  padding: 0 12px;
}

.jp-Shortcuts-Symbols td:nth-child(2) {
  padding-right: 10px;
}

.jp-Shortcuts-Search {
  height: 30px;
}

.jp-Shortcuts-Search > input {
  box-shadow: none;
}

.jp-Shortcuts-AdvancedOptionsSmall {
  width: 30%;
}

.jp-Shortcuts-AdvancedOptionsRight {
  margin-top: 8px;
}

.jp-Shortcuts-AdvancedOptionsLink {
  color: var(--jp-content-link-color);
  text-decoration: none;
  margin-right: 15px;
}

.jp-Shortcuts-AdvancedOptionsLink:hover {
  color: var(--jp-brand-color0);
}

.jp-Shortcuts-AdvancedOptionsLink:active {
  color: var(--jp-brand-color0);
}

.jp-Shortcuts-HeaderRowContainer {
  padding-right: 14px;
}

.jp-Shortcuts-HeaderRow {
  font-weight: bold;
  font-size: var(--jp-ui-font-size1);
  background-color: var(--jp-layout-color0);
  width: 100%;
  z-index: 1;
  display: table;
  padding: 10px 0;
}

.jp-Shortcuts-commandIcon {
  margin-right: 13px;
}

.jp-Shortcuts-altIcon {
  margin-right: 14px;
}

.jp-Shortcuts-controlIcon {
  margin-left: 8px;
  margin-right: 16px;
}
</style><style>/**
 * Copyright (c) 2014 The xterm.js authors. All rights reserved.
 * Copyright (c) 2012-2013, Christopher Jeffrey (MIT License)
 * https://github.com/chjj/term.js
 * @license MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * Originally forked from (with the author's permission):
 *   Fabrice Bellard's javascript vt100 for jslinux:
 *   http://bellard.org/jslinux/
 *   Copyright (c) 2011 Fabrice Bellard
 *   The original design remains. The terminal itself
 *   has been extended to include xterm CSI codes, among
 *   other features.
 */

/**
 *  Default styles for xterm.js
 */

.xterm {
    cursor: text;
    position: relative;
    user-select: none;
    -ms-user-select: none;
    -webkit-user-select: none;
}

.xterm.focus,
.xterm:focus {
    outline: none;
}

.xterm .xterm-helpers {
    position: absolute;
    top: 0;
    /**
     * The z-index of the helpers must be higher than the canvases in order for
     * IMEs to appear on top.
     */
    z-index: 5;
}

.xterm .xterm-helper-textarea {
    padding: 0;
    border: 0;
    margin: 0;
    /* Move textarea out of the screen to the far left, so that the cursor is not visible */
    position: absolute;
    opacity: 0;
    left: -9999em;
    top: 0;
    width: 0;
    height: 0;
    z-index: -5;
    /** Prevent wrapping so the IME appears against the textarea at the correct position */
    white-space: nowrap;
    overflow: hidden;
    resize: none;
}

.xterm .composition-view {
    /* TODO: Composition position got messed up somewhere */
    background: #000;
    color: #FFF;
    display: none;
    position: absolute;
    white-space: nowrap;
    z-index: 1;
}

.xterm .composition-view.active {
    display: block;
}

.xterm .xterm-viewport {
    /* On OS X this is required in order for the scroll bar to appear fully opaque */
    background-color: #000;
    overflow-y: scroll;
    cursor: default;
    position: absolute;
    right: 0;
    left: 0;
    top: 0;
    bottom: 0;
}

.xterm .xterm-screen {
    position: relative;
}

.xterm .xterm-screen canvas {
    position: absolute;
    left: 0;
    top: 0;
}

.xterm .xterm-scroll-area {
    visibility: hidden;
}

.xterm-char-measure-element {
    display: inline-block;
    visibility: hidden;
    position: absolute;
    top: 0;
    left: -9999em;
    line-height: normal;
}

.xterm.enable-mouse-events {
    /* When mouse events are enabled (eg. tmux), revert to the standard pointer cursor */
    cursor: default;
}

.xterm.xterm-cursor-pointer,
.xterm .xterm-cursor-pointer {
    cursor: pointer;
}

.xterm.column-select.focus {
    /* Column selection mode */
    cursor: crosshair;
}

.xterm .xterm-accessibility,
.xterm .xterm-message {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    z-index: 10;
    color: transparent;
}

.xterm .live-region {
    position: absolute;
    left: -9999px;
    width: 1px;
    height: 1px;
    overflow: hidden;
}

.xterm-dim {
    opacity: 0.5;
}

.xterm-underline-1 { text-decoration: underline; }
.xterm-underline-2 { text-decoration: double underline; }
.xterm-underline-3 { text-decoration: wavy underline; }
.xterm-underline-4 { text-decoration: dotted underline; }
.xterm-underline-5 { text-decoration: dashed underline; }

.xterm-strikethrough {
    text-decoration: line-through;
}

.xterm-screen .xterm-decoration-container .xterm-decoration {
	z-index: 6;
	position: absolute;
}

.xterm-decoration-overview-ruler {
    z-index: 7;
    position: absolute;
    top: 0;
    right: 0;
    pointer-events: none;
}

.xterm-decoration-top {
    z-index: 2;
    position: relative;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Terminal {
  min-width: 240px;
  min-height: 120px;
}

.jp-Terminal-body {
  padding: 8px;
}

[data-term-theme='inherit'] .xterm .xterm-screen canvas {
  border: 1px solid var(--jp-layout-color0);
}

[data-term-theme='light'] .xterm .xterm-screen canvas {
  border: 1px solid #fff;
}

[data-term-theme='dark'] .xterm .xterm-screen canvas {
  border: 1px solid #000;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-toc-numberingButton:hover button.lm-mod-toggled.jp-Button.jp-mod-minimal {
  background-color: var(--jp-inverse-layout-color4);
}

.jp-toc-numberingButton button.lm-mod-toggled .jp-icon3[fill] {
  fill: var(--jp-layout-color1);
}

.jp-toc-numberingButton button.lm-mod-toggled.jp-Button.jp-mod-minimal {
  background-color: var(--jp-inverse-layout-color3);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Tooltip {
  background: var(--jp-layout-color1);
  border: var(--jp-border-width) solid var(--jp-border-color1);
  font-size: var(--jp-ui-font-size0);
  box-shadow: var(--jp-elevation-z6);
  max-width: 750px;
  max-height: 350px;
  z-index: 10001;
  padding: 4px;
  display: flex;
}

.jp-Tooltip-content {
  overflow: auto;
}

.jp-Tooltip:focus {
  outline: 0;
}

.jp-Tooltip pre {
  white-space: pre-wrap;
  margin: 0;
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-RenderedVegaCommon5 {
  margin-left: 8px;
  margin-top: 8px;
}

.jp-MimeDocument .jp-RenderedVegaCommon5 {
  padding: 16px;
}

.vega canvas {
  background: var(--jp-vega-background);
}
</style><script async="" src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/remoteEntry.36690a2bfa9ee0dd7bc8.js.download"></script><script async="" src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/remoteEntry.83218267865631694ab2.js.download"></script><script async="" src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/remoteEntry.235a4eb5f08bc399672c.js.download"></script><script async="" src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/remoteEntry.3318e43690692165b5ed.js.download"></script><script async="" src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/remoteEntry.ec403c961f3e6d85e7bc.js.download"></script><script async="" src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/remoteEntry.0dca0b46350f0c25c608.js.download"></script><script async="" src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/remoteEntry.ca1efc27dc965162ca86.js.download"></script><script async="" src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/remoteEntry.cfbccb113f4c7c1d65fe.js.download"></script><script async="" src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/remoteEntry.c8393541423362724be3.js.download"></script><style>/**
  Copyright (c) Jupyter Development Team.
  Distributed under the terms of the Modified BSD License.
*/

/* Add CSS variables to :root */
:root {
  --jp-icon-plotly: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 132 132'%3E %3Cdefs%3E %3Cstyle%3E .cls-1 %7B fill: %23119dff; %7D .cls-2 %7B fill: %2325fefd; %7D .cls-3 %7B fill: %23fff; %7D %3C/style%3E %3C/defs%3E %3Ctitle%3Eplotly-logomark%3C/title%3E %3Cg id='symbol'%3E %3Crect class='cls-1' width='132' height='132' rx='6' ry='6'/%3E %3Ccircle class='cls-2' cx='78' cy='54' r='6'/%3E %3Ccircle class='cls-2' cx='102' cy='30' r='6'/%3E %3Ccircle class='cls-2' cx='78' cy='30' r='6'/%3E %3Ccircle class='cls-2' cx='54' cy='30' r='6'/%3E %3Ccircle class='cls-2' cx='30' cy='30' r='6'/%3E %3Ccircle class='cls-2' cx='30' cy='54' r='6'/%3E %3Cpath class='cls-3' d='M30,72a6,6,0,0,0-6,6v24a6,6,0,0,0,12,0V78A6,6,0,0,0,30,72Z'/%3E %3Cpath class='cls-3' d='M78,72a6,6,0,0,0-6,6v24a6,6,0,0,0,12,0V78A6,6,0,0,0,78,72Z'/%3E %3Cpath class='cls-3' d='M54,48a6,6,0,0,0-6,6v48a6,6,0,0,0,12,0V54A6,6,0,0,0,54,48Z'/%3E %3Cpath class='cls-3' d='M102,48a6,6,0,0,0-6,6v48a6,6,0,0,0,12,0V54A6,6,0,0,0,102,48Z'/%3E %3C/g%3E %3C/svg%3E");
}

/* Base styles */
.jp-RenderedPlotly {
  width: 100%;
  height: 100%;
  padding: 0;
  overflow: hidden;
}

/* Document styles */
.jp-MimeDocument .jp-RenderedPlotly {
  overflow: hidden;
}

/* Document icon */
.jp-PlotlyIcon {
  background-image: var(--jp-icon-plotly);
}
</style><style>/*
    See the JupyterLab Developer Guide for useful CSS Patterns:

    https://jupyterlab.readthedocs.io/en/stable/developer/css.html
*/
</style><style>/*
    See the JupyterLab Developer Guide for useful CSS Patterns:

    https://jupyterlab.readthedocs.io/en/stable/developer/css.html
*/
:root {
  --as-main-bg-color-primary: var(--jp-layout-color0);
  --as-main-bg-color-secondary: var(--jp-layout-color1);
  --as-main-bg-color-third: var(--jp-layout-color2);
  --as-main-font-color-black: var(--jp-ui-font-color0);
  --as-main-font-color: var(--jp-ui-font-color1);
  --as-main-font-color-secondary: var(--jp-ui-font-color2);
  --as-main-disabled-color: var(--jp-ui-font-color3);
  --as-text-color-hover: var(--jp-input-hover-background);
  --as-background-hover: var(--jp-input-background);
  --as-border-color: var(--jp-border-color);
  --as-border-color-secondary: var(--jp-border-color1);
  --as-main-font-color-inverse: var(--jp-ui-inverse-font-color0);
  --as-main-font-color-gray-inverse: var(--jp-ui-inverse-font-color3);
  --as-main-bg-color-primary-inverse: var(--jp-inverse-layout-color0);
  --as-main-bg-color-gray-inverse: var(--jp-inverse-layout-color4);
  --as-border-color-light: var(--jp-border-color2);
}

/* @TODO remove this in the future */
.anaconda-assistant-tooltip {
  background-color: var(--as-main-bg-color-primary-inverse) !important;
  color: var(--as-main-font-color-inverse) !important;
}

.as-notebook-container #jp-right-stack, .as-notebook-container #jp-left-stack {
  min-width: 480px !important;
}

.as-widget-container {
  min-width: 480px !important;
  background: #f5f5f5;
  height: 100%;
}

.as-widget-content {
  height: 100%;
  background: #fff;
  color: #000;
}

.as-container {
  background: var(--as-main-bg-color-secondary);
  border-radius: 4px;
  color: var(--as-main-font-color);
  display: flex;
  flex-direction: column;
  font-family: 'Inter', sans-serif;
  height: 100%;
  overflow: auto;
  width: 100%;
}

.as-login-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
  margin: 50px 50px;
  background: var(--as-main-bg-color-secondary);
  color: var(--as-main-font-color);
  height: 100%;
  font-family: 'Inter', sans-serif;
}

.as-login-container a {
  color: #6c5af6;
  text-decoration: underline;
  cursor: pointer;
}

.as-login-header {
  display: flex;
  margin-bottom: 40px;
  color: var(--as-main-font-color);
}

.as-login-body,
.as-login-footer {
  margin-left: 15px;
}

.as-login-body {
  flex-grow: 1;
  font-size: 16px;
  font-weight: 500;
  line-height: 24px;
}

.as-login-body ul {
  margin: 20px 0;
  padding: 0;
  list-style: none;
}

.as-login-body ul li {
  display: flex;
  margin-bottom: 15px;
  align-items: center;
}

.as-login-body ul li * {
  fill: #0cca4a;
}

.as-login-body ul li.coming-soon {
  color: #6c5af6;
  margin-top: 35px;
}

.as-login-body ul li.disabled,
.as-login-body ul li.disabled * {
  color: var(--as-main-disabled-color);
  fill: var(--as-main-disabled-color);
}

.as-create-account-button {
  margin-bottom: 20px;
  margin-top: 20px;
  border-radius: 4px;
  padding: 10px 16px;
  background: #0cca4a;
  border: none;
  color: #06262d;
  cursor: pointer;
  font-size: 16px;
  font-weight: 700;
  line-height: 24px;
}

.as-create-account-button:hover {
  background: #4fab4f;
}

.as-login-sign-in-text {
  font-size: 14px;
  font-weight: 400;
  line-height: 17px;
}

.as-login-footer {
  font-size: 16px;
  font-weight: 500;
  line-height: 20px;
  display: flex;
  margin-bottom: 60px;
}

.as-login-footer svg,
.as-login-body svg {
  width: 17px;
  height: 17px;
  margin-right: 10px;
}

.as-login-footer svg {
  width: 20px;
  height: 20px;
}
.as-login-icon {
  min-width: 30px;
  max-width: 30px;
  color: #31a824;
}

.as-title-text {
  font-size: 24px;
  font-weight: 700;
  margin-left: 10px;
  font-family: 'Inter', sans-serif;
}

.as-subtitle-text {
  font-size: 16px;
  font-weight: 400;
  line-height: 24px;
}

.as-home-screen-container {
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.as-home-screen-icon {
  width: 80px;
  color: #31a824;
}

.as-home-screen-df-icon-container {
  width: 26px;
  height: 26px;
  background: #e9e6fe;
  border-radius: 100px;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-right: 10px;
}

.as-home-screen-df-icon {
  width: 12px;
  height: 12px;
  color: #6c5af6;
}

.as-version-text {
  color: #bdbdbd;
  font-size: 11px;
  margin-top: 4px;
  margin-left: 10px;
}

.as-no-dataframe-container {
  display: flex;
  flex-direction: column;
  background: #f7f7f7;
  border-radius: 4px;
  padding: 20px 10px;
}

.as-no-dataframe-heading {
  font-size: 14px;
  font-weight: lighter;
  margin-bottom: 20px;
}

.as-no-dataframe-text {
  font-size: 12px;
  margin-bottom: 10px;
}

.as-no-dataframe-link {
  display: inline;
  color: blue;
  text-decoration: underline;
}

.as-no-dataframe-link:hover {
  color: darkblue;
  text-decoration: underline;
}

/*Main Chat Screen*/

.as-chat-main-section {
  overflow-y: scroll;
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 16px;
}

.as-chat-message-container {
  display: flex;
  flex-direction: column;
}

.as-export-chat-container {
  display: flex;
  align-items: center;
  padding: 0 20px;
  min-height: 26px;
  justify-content: center;
  color: #6c5af6;
  gap: 10px;
  cursor: pointer;
  margin-top: auto;
  font-size: 16px;
  font-style: normal;
  font-weight: 400;
  line-height: 20px;
}

.as-export-chat-container svg {
  height: 12px;
}

@keyframes ellipsis {
  to {
    width: 1.25em;
  }
}

@-webkit-keyframes ellipsis {
  to {
    width: 1.25em;
  }
}

.as-chat-footer-container {
  border-top: 3px solid #ced2d9;
  background: var(--as-background-hover);
  padding: 16px 10px;
  z-index: 9;
}

.as-chat-footer-container .anaconda-assistant-chat-input {
  margin: 0 6px 8px 6px;
}

.as-chat-footer-container .anaconda-assistant-chat-footer {
  margin: 0 18px;
}

#as-arrow,
#as-arrow::before {
  position: absolute;
  width: 8px;
  height: 8px;
  background: #caecca;
}

#as-arrow {
  visibility: hidden;
}

#as-arrow::before {
  visibility: visible;
  background: #caecca;
  content: '';
  transform: rotate(45deg);
  bottom: 4px;
}

.as-settings-container {
  display: flex;
  flex-direction: column;
  margin-top: 20px;
  margin-bottom: 40px;
  overflow-y: scroll;
}

.as-setting-control-container {
  border-bottom: 1px solid rgba(85, 85, 85, 0.05);
  padding: 20px;
  height: -webkit-fill-available;
  overflow-y: scroll;
}

.as-setting-control-title {
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 10px;
}

.as-setting-control-inner-container {
  display: flex;
  align-items: center;
  justify-content: end;
  gap: 20px;
  font-size: 13px;
  margin: 20px 0;
}

.as-setting-control-description {
  max-width: 77%;
}

.as-settings-template-container {
  padding-top: 20px;
}

.as-settings-sub-header-container {
  font-size: 12px;
  font-weight: lighter;
  color: #9ca3af;
  margin-bottom: 25px;
  display: flex;
  flex-direction: column;
}

.as-setting-container {
  display: flex;
  flex-direction: column;
  margin-bottom: 20px;
}

.as-setting-control-description.as-setting-control-description-context-limit {
  max-width: 100%;
  flex-grow: 1;
}

.as-setting-control-description.as-setting-control-description-context-limit svg {
  width: 12px;
  margin-right: 4px;
  display: inline-block;
}

.as-setting-control-description.as-setting-control-description-context-limit
  .as-code-upgrade-button {
  align-items: center;
  background: #4fab4f;
  border-radius: 4px;
  border: none;
  color: white;
  cursor: pointer;
  display: flex;
  font-size: 16px;
  font-style: normal;
  justify-content: center;
  margin-bottom: 5px;
  margin-top: 15px;
  margin-left: auto;
  margin-right: 20px;
  padding: 8px;
}

.as-setting-control-description.as-setting-control-description-context-limit
  .as-code-upgrade-button:hover {
  background: #0cca4a;
  color: #06262d;
}

.as-setting-control-description.as-setting-logout {
  max-width: 100%;
}

.as-setting-control-description .as-logout-button {
  color: #444;
  align-items: center;
  border-radius: 4px;
  border: none;
  cursor: pointer;
  display: flex;
  font-size: 16px;
  font-style: normal;
  justify-content: center;
  padding: 8px;
}

.as-setting-control-description.as-setting-control-description-context-limit a {
  color: #6c5af6;
  text-decoration: underline;
}

.as-access-inner-container {
  padding: 0px 20px 0px 20px;
}

.as-access-inner-container-buttons {
  padding: 0px 20px 0px 20px;
  display: flex;
  justify-content: space-between;
}

.as-access-header {
  font-size: 24px;
  font-weight: lighter;
  margin-bottom: 10px;
  margin-top: 10px;
  font-family: 'Inter', sans-serif;
}

.as-access-text {
  font-size: 13px;
  font-weight: normal;
}

.as-access-inner-container a {
  color: #6c5af6;
  text-decoration: underline;
}

.as-terms-button,
.as-access-button,
.as-access-inner-container-buttons .as-logout-button {
  margin-top: 35px;
  border-radius: 4px;
  padding: 12px;
  background: #4fab4f;
  border: none;
  color: white;
  cursor: pointer;
  font-size: 16px;
}

.as-terms-button {
  margin-left: auto;
}

.as-terms-button:hover {
  background: #0cca4a;
  color: black;
}

.as-terms-error-button {
  border-radius: 4px;
  padding: 12px;
  background: #3874cb;
  border: none;
  color: white;
  cursor: pointer;
}

.as-terms-error {
  border-radius: 6px;
  /* Transparent red */
  background-color: rgba(255, 0, 0, 0.1);
  padding: 10px;
  font-size: 12px;
}

.as-terms-error-text {
  color: #ff0000;
  font-size: 12px;
  margin-bottom: 10px;
}

.as-back-button-container {
  height: 30px;
  padding: 0 18px;
  display: flex;
  align-items: center;
  color: #778;
  margin-bottom: 10px;
}

.as-back-button-icon {
  width: 14px;
  padding-top: 4px;
  padding-left: 2px;
  padding-right: 2px;
  color: #6c5af6;
  cursor: pointer;
}

.as-back-button-title-container {
  width: 100%;
  display: grid;
  text-align: center;
}

.as-back-button-title {
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}

.as-popup-outer-container {
  position: fixed;
  width: 480px;
  right: 32px;
  bottom: 80px;
}

.as-explain-popup {
  position: absolute;
  border: 1px solid rgba(224, 224, 224, 0.8);
  box-shadow: 0 0 16px rgba(0, 0, 0, 0.1);
  border-radius: 8px;
  background: white;
  z-index: 1;
}

.as-popup-container {
  padding: 0 16px;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.as-popup-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 13px;
  font-weight: bold;
  padding-top: 10px;
  padding-bottom: 10px;
  color: #212121;
}

.as-popup-logo {
  display: flex;
  align-items: center;
  color: #2c303a;
  font-style: normal;
  font-weight: 700;
  font-size: 14px;
}

.as-popup-icon-container {
  background: #4fab4f;
  border-radius: 4px;
  margin-right: 10px;
  padding: 8px 4px 5px 4px;
}

.as-popup-icon {
  width: 20px;
  height: 16px;
  color: white;
}

.as-popup-close-icon {
  width: 14px;
  color: rgb(33, 33, 33, 0.5);
  cursor: pointer;
}

.as-popup-close-icon:hover {
  color: rgb(33, 33, 33, 0.8);
}

.as-popup-buttons {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding-bottom: 15px;
}

.as-popup-button {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: white;
  border: 1px solid #e6e6e6;
  margin-bottom: 3px;
  border-radius: 8px;
  padding: 9px 12px 9px 12px;
  cursor: pointer;
}

.as-popup-button:hover {
  background: rgba(33, 33, 33, 0.05);
}

.as-popup-button-icon {
  width: 8px;
  height: 12px;
  color: rgba(33, 33, 33, 0.5);
}

.as-popup-buttons-section {
  display: flex;
  align-items: center;
  font-size: 13px;
  font-weight: 590;
  letter-spacing: 0.02em;
}
</style><style>@font-face {
  font-family: "Inter";
  font-style: normal;
  font-weight: 100;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/2ab12bc9c828b7d0441b.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: italic;
  font-weight: 100;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/e5555942ce99d0b615e8.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: normal;
  font-weight: 200;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/b6c261b852a5d2f8dec6.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: italic;
  font-weight: 200;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/47608eb39ed9fef3239c.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: normal;
  font-weight: 300;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/d12f464576228dc3879b.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: italic;
  font-weight: 300;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/72319cf2f5403460dd70.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/b6f9db9e45be20f3c131.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: italic;
  font-weight: 400;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/708b7baa59a80d804652.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: normal;
  font-weight: 500;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/8458f8afa67b5691c1fc.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: italic;
  font-weight: 500;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/9f4481b062ed1cf93cd7.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: normal;
  font-weight: 600;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/8e52a861dc26ff4608c5.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: italic;
  font-weight: 600;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/8789e3e4f2cb727ad4ce.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: normal;
  font-weight: 700;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/220976705fbec109f43c.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: italic;
  font-weight: 700;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/af46bd584de8bb0aabfd.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: normal;
  font-weight: 800;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/98023dd780794f5f7f58.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: italic;
  font-weight: 800;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/fe47bb9ab34536443235.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: normal;
  font-weight: 900;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/aa867fe6d7113966e6ed.woff2?v=4.0) format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: italic;
  font-weight: 900;
  font-display: swap;
  src: url(http://localhost:8888/lab/extensions/@anaconda/assistant/static/4b50bf8326f73ec4636c.woff2?v=4.0) format("woff2");
}
</style><style>/*
    See the JupyterLab Developer Guide for useful CSS Patterns:

    https://jupyterlab.readthedocs.io/en/stable/developer/css.html
*/
</style><style>/*
    See the JupyterLab Developer Guide for useful CSS Patterns:

    https://jupyterlab.readthedocs.io/en/stable/developer/css.html
*/
</style><style>.jp-PublishAppForm {
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  font-size: 16px;
  display: flex;
  flex-direction: column;
  align-items: start;
  justify-content: start;
  text-align: start;
  padding: 12px 20px;
  width: -webkit-fill-available;
}

.jp-PythonAnywhereIcon svg > g > *,
.jp-app-menu-option-icon svg > path {
  fill: var(--jp-inverse-layout-color3);
}

.jp-PublishAppForm label {
  font-style: normal;
  font-weight: 700;
  font-size: 16px;
  margin-bottom: 10px;
}

.jp-PublishAppForm h1 {
  font-weight: 400;
  font-size: 20px;
  padding-bottom: 10px;
}

.jp-PublishAppForm-text-gray {
  font-weight: 400;
  color: var(--jp-ui-font-color2);
  margin-bottom: 10px;
}

.jp-flex-row {
  display: flex;
  flex-direction: row;
  margin-bottom: 10px;
  width: inherit;
}

.jp-file-info {
  margin: 10px;
  width: inherit;
}

.jp-PublishAppForm input,
textarea {
  width: -webkit-fill-available;
  margin-bottom: 0.5em;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
  background: var(--jp-input-background);
  border-radius: 4px;
}

.jp-validation-error-label-color {
  color: #d00000;
}

.jp-validation-error-input-border {
  border: 1px solid #d00000 !important;
}

.jp-PublishAppForm input {
  height: 34px;
}

.jp-PublishAppForm textarea {
  height: 96px;
}

.jp-text-small {
  font-size: 11px;
}

.jp-PublishAppForm-text-validation {
  color: var(--jp-ui-font-color2);
  font-size: 10px;
  width: -webkit-fill-available;
  text-align: right;
}

.jp-PublishAppForm-button-div {
  display: flex;
  width: 100%;
  flex-direction: row;
  justify-content: flex-end;
}

.jp-PublishAppForm-button-div-right,
.jp-PublishAppForm-button-div-left {
  margin-top: 20px;
  display: flex;
  flex-direction: row;
  gap: 10px;
}

.jp-PublishAppForm-button-div-right {
  align-self: flex-end;
}

.jp-PublishAppForm-button-div-left {
  align-self: flex-start;
}

.jp-PublishAppForm-text-validation-error {
  color: #d00000;
  font-weight: 400;
  font-size: 12px;
  margin-top: -4px;
  margin-bottom: 8px;
}

.jp-PublishAppForm-button {
  padding: 10px 18px;
  border-radius: 8px;
  cursor: pointer;
}

.jp-PublishAppForm-button-icon {
  margin-right: 10px;
  font-size: 12px;
}

.jp-PublishAppForm-version {
  background: rgba(67, 176, 73, 0.12);
  opacity: 1;
  border-radius: 8px;
  padding: 5px;
  color: #000;
  margin-bottom: 10px;
}

.jp-help-button {
  position: relative;
  cursor: pointer;
  height: 40px;
  width: 40px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.jp-help-button:hover,
.jp-help-button:active,
.jp-help-button-active {
  width: 40px;
  height: 40px;
  background-color: #e5e8eb;
  border-radius: 4px;
}

.jp-cancel-btn {
  color: var(--jp-ui-font-color2) !important;
  border: none !important;
  background: none !important;
  cursor: pointer;
}

.jp-cancel-btn:hover {
  color: #52596b !important;
  background-color: #f9fafb !important;
}

.jp-primary-btn {
  color: #2c303a;
  background: #0ccb4a;
  border: 1px solid #0ccb4a;
  font-weight: 700;
}

.jp-primary-btn:hover {
  background: #4fab4f;
  border: 1px solid #4fab4f;
}

.jp-primary-btn__error {
  border: 1px solid #d00000;
  color: #d00000;
  font-weight: 700;
  background: none;
}

.jp-secondary-btn__error {
  border: 1px solid #d00000;
  color: #d00000;
  font-weight: 700;
  background: none;
}

.jp-primary-btn__error:hover,
.jp-secondary-btn__error:hover {
  border: 1px solid #b42318;
  color: #b42318;
  background-color: #f9fafb;
}

.jp-primary-btn:disabled {
  background: rgb(67 176 73 / 50%);
  color: #fff;
  border: none;
  cursor: not-allowed;
}

.jp-secondary-btn {
  border-radius: 4px;
  border: 1px solid #06262D !important;
  background: var(--bg-color, #FFF);
}

.jp-secondary-btn:hover {
  border: 1px solid #06262D;
  color: #000;
}

.jp-secondary-btn:disabled {
  background: #fff;
  border: 1px solid rgb(67 176 73 / 30%);
  cursor: not-allowed;
  color: #43b049;
}

.jp-tertiary-btn {
  border-radius: 4px;
  border: 1px solid var(--informational-error, #D00000);
  background: var(--informational-error, #D00000);
  box-shadow: 0px 1px 2px 0px rgba(16, 24, 40, 0.05);
  color: #fff !important;
}

.jp-tertiary-btn:hover {
  color: #ced2d9;
}

.jp-delete-btn {
  background-color: #d00000 !important;
  border: 1px solid #d00000;
  box-shadow: 0 1px 2px rgb(16 24 40 / 5%);
  border-radius: 8px;
  cursor: pointer;
}

.jp-delete-btn:hover {
  background-color: #b42318 !important;
  border: #b42318 !important;
}

.jp-delete-btn:focus,
.jp-cancel-btn:focus {
  outline: none !important;
}

.jp-disable-secondary-btn img {
  vertical-align: bottom;
  padding-right: 5px;
}

.jp-confirmation-alert {
  text-align: left;
  max-width: 300px;
}

.jp-PAManager {
  display: flex;
  flex-direction: column;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size1);
}

.jp-PAManager-section {
  display: flex;
  flex: 0 1 auto;
  flex-direction: column;
  height: 100%;
  justify-content: space-between;
  overflow: auto;
}

.jp-PAManager-section h1 {
  flex: 0 0 auto;
  font-weight: 600;
  letter-spacing: 1px;
  font-size: 17px;
  margin: 0;
}

.jp-PAManager-sectionHeader .jp-ToolbarButtonComponent {
  flex: 0 0 auto;
}

.jp-PAManager-sectionContainer {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  overflow: auto;
}

.jp-PAManager-section-header {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color2);
}

.jp-PAManager-section-header h1 {
  padding: 15px;
}

.jp-PAManager-section-body {
  padding: 5px 15px;
}

.jp-PAManager-section-body p {
  padding: 10px 0;
}

.jp-PAManager-section-footer {
  margin: 24px;
}

.jp-PAManager-sectionList {
  margin: 0;
  padding: 0 15px;
  list-style-type: none;
  padding-inline-start: 0;
}

.jp-PAManager-sectionList h2 {
  font-weight: 700 !important;
  font-size: 16px !important;
}

.jp-PAManager-listItem {
  display: flex;
  flex-direction: column;
  color: var(--jp-ui-font-color1);
  margin-bottom: 14px;
}

.jp-PAManager-listItem-flex {
  display: flex;
  padding: 6px;
  margin-bottom: 3px;
}

.jp-PAManager-listItem-project:hover,
.jp-highlight-text {
  background: var(--jp-brand-color4);
  opacity: 1;
  color: #2c303a;
  cursor: pointer;
}

.jp-hidden {
  visibility: hidden;
}

.jp-PAManager-listItem-version {
  margin-left: 0.5em;
  font-weight: 400;
  font-size: 10px;
  background: var(--jp-brand-color4);
  opacity: 1;
  color: #000;
  border-radius: 8px;
  padding: 3px 5px;
  text-align: center;
}

.jp-PAManager-listItem-project:hover .jp-PAManager-listItem-version {
  background: #fff;
}

.jp-PAManager-listItem-title {
  flex: 1 1 70%;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

.jp-file-name {
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

.jp-PAManager-listItem-statusSection {
  padding: 0 10px 0 0;
}

.jp-PAManager-listItem-status {
  margin-left: 0.7em;
}

.jp-PAManager-SettingsIcon {
  margin-left: 0.5em;
  cursor: pointer;
  filter: contrast(0.5);
}

.jp-PAManager-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-app-menu {
  border: 1px solid rgb(0 0 0 / 87%);
  box-shadow: 0 4px 4px rgb(0 0 0 / 25%);
}

.jp-app-menu-option-icon svg {
  vertical-align: middle;
}

.jp-app-status-container {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  text-align: center;
  width: 100%;
  color: var(--jp-ui-font-color0);
}

.jp-app-status-icon img {
  width: 85px;
}

.jp-mt-40 {
  margin-top: 40px;
}

.jp-app-status-desc {
  margin: 40px;
  text-align: left;
  word-break: break-all;
}

.jp-app-status-desc a {
  color: #34d;
}

.jp-PublishAppForm-text-panel-warning {
  font-style: normal;
  font-weight: 400;
  font-size: 14px;
  margin-top: 10px;
  margin-bottom: 0;
  color: var(--informational-error, #D00000);
}


.jp-PublishAppForm-code {
  background: lightgrey;
  color: var(--jp-ui-font-color0);
  border-radius: 4px;
  padding: 2px 4px;
}

.jp-PublishAppForm-text-legal-info {
  font-style: normal;
  font-weight: 400;
  font-size: 11px;
  margin-top: 20px;
  margin-bottom: 0px;
  color: var(--jp-ui-font-color0);
}


.jp-PublishAppForm-text-panel-tip {
  font-style: normal;
  font-weight: 400;
  font-size: 16px;
  margin-top: 20px;
  margin-bottom: 0px;
  color: var(--jp-ui-font-color1);
}


.jp-PublishAppForm-text-legal-info a,
.jp-PAManager-section-body a {
  text-decoration: underline;
  color: #106ba3;
  cursor: pointer;
}

/* visited status */
.jp-PublishAppForm-text-legal-info a:visited {
  color: #551a8b;
}

.jp-app-status-actions {
  position: absolute;
  right: 30px;
  bottom: 30px;
}

.jp-PAManager-itemLabel {
  font-size: var(--jp-ui-font-size1);
  flex: 1 1 55%;
  padding: 0 4px;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

.jp-PAManager-itemLabel:focus {
  background-color: var(--jp-layout-color2);
}

.jp-PAManager-itemDetail {
  font-size: var(--jp-ui-font-size1);
  flex: 1 1 45%;
  padding: 0 4px;
  text-overflow: ellipsis;
  overflow: hidden;
  white-space: nowrap;
}

.jp-page-not-found {
  position: absolute;
  padding: 40px 20% 60px;
  background: #06262d;
  display: flex;
  flex-direction: column;
  color: #fff;
  height: 100%;
  width: -webkit-fill-available;
  background-repeat: no-repeat;
  background-size: auto 80%;
  background-position: center bottom;
}

.jp-page-not-found h1 {
  font-size: 28px;
  font-weight: 400;
}

.jp-page-not-found h2 {
  font-size: 19px;
  font-weight: 400;
}

.jp-link {
  color: #43b049 !important;
}

.jp-page-not-found-actions {
  display: flex;
  column-gap: 30px;
}

.jp-mb40 {
  margin-bottom: 40px;
  margin-top: 0;
}

.jp-PAManager-item .jp-PAManager-itemShutdown {
  border-radius: 0;
}

.jp-PAManager-item:not(:hover) .jp-PAManager-itemShutdown {
  visibility: hidden;
}

.jp-PAManager-sectionList
  .jp-PAManager-item
  .jp-Button.jp-PAManager-itemShutdown:hover {
  background: var(--jp-layout-color3);
}

.jp-PAManager-shutdownAll.jp-mod-styled {
  margin: 0 8px;
  color: var(--jp-warn-color1);
  background-color: transparent;
  height: var(--jp-private-running-shutdown-button-height);
  line-height: var(--jp-private-running-shutdown-button-height);
  border-radius: 2px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-PAManager-shutdownAll.jp-mod-styled:hover {
  background-color: var(--jp-layout-color2);
}

.jp-PAManager-shutdownAll.jp-mod-styled:focus {
  border: none;
  box-shadow: none;
  background-color: var(--jp-layout-color2);
}

.jp-PAManager-shutdownAll.jp-mod-styled.jp-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.jp-PAManager-shutdownAll.jp-mod-styled.jp-mod-disabled:hover {
  background: none;
}

.jp-tooltip {
  max-width: 185px;
  line-height: 18px;
}

#app-quota {
  max-width: 90px;
  cursor: pointer;
}

.jp-tooltip__limit {
  color: #d00000;
}

.jp-xterm-container {
  width: 100%;
  height: 100%;
}

.jp-xterm-loading-icon {
  position: absolute;
  margin: auto;
  inset: 0;
  z-index: 1;
}

.jp-NotebookPanel-notebook {
  min-width: 300px;
}

.jp-CustomWidget {
  min-width: 300px;
  overflow: auto;
}

.loading-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  width: -webkit-fill-available;
}

.jp-app-menu-option:hover {
  background-color: #0cca4a;
}

.anaconda-logo-icon {
  align-items: center;
  display: flex;
  height: 52px;
  width: 52px;
  background-color: var(--anaconda-primary-color);
  -webkit-mask-image: var(--anaconda-icon-launcher);
  mask-image: var(--anaconda-icon-launcher);
}

/* New Design Changes */
/* @TODO put on top the root block */
:root {
  --anaconda-rs-running-icon: url("data:image/svg+xml,%3csvg width='16' height='10' viewBox='0 0 16 10' fill='none' xmlns='http://www.w3.org/2000/svg' %3e %3ccircle cx='11' cy='5' r='3' fill='%23FFBA0A'%3e %3canimate attributeName='r' from='3' to='2' dur='1s' repeatCount='indefinite' /%3e %3c/circle%3e %3ccircle cx='4' cy='5' r='2' fill='%23FFBA0A'%3e %3canimate attributeName='r' from='2' to='3' dur='1s' repeatCount='indefinite' /%3e %3c/circle%3e %3c/svg%3e");
  --anaconda-rs-check-icon: url("data:image/svg+xml,%3csvg width='16' height='16' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath fill-rule='evenodd' clip-rule='evenodd' d='M1 8C1 4.13437 4.13437 1 8 1C11.8656 1 15 4.13437 15 8C15 11.8656 11.8656 15 8 15C4.13437 15 1 11.8656 1 8ZM3 8L7 12L13 6L11.5938 4.59375L7 9.1875L4.40625 6.59375L3 8Z' fill='%2327871D'/%3e %3c/svg%3e");
  --anaconda-rs-warning-icon: url("data:image/svg+xml,%3csvg width='16' height='16' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg id='warning'%3e %3cpath id='vector' fill-rule='evenodd' clip-rule='evenodd' d='M15 8C15 11.8656 11.8656 15 8 15C4.13437 15 1 11.8656 1 8C1 4.13437 4.13437 1 8 1C11.8656 1 15 4.13437 15 8ZM14 8C14 4.6875 11.3125 2 8 2C4.6875 2 2 4.6875 2 8C2 11.3125 4.6875 14 8 14C11.3125 14 14 11.3125 14 8ZM7 9V4H9V9H7ZM7 10V12H9V10H7Z' fill='%23DC3246'/%3e %3c/g%3e %3c/svg%3e");
  --anaconda-icon-launcher: url("data:image/svg+xml,%3csvg version='1.1' id='Layer_1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' x='0px' y='0px' viewBox='0 0 512 512' style='enable-background:new 0 0 512 512%3b' xml:space='preserve'%3e %3cg%3e %3cpath d='M93%2c391.4c-19.3-1.9-38.5-5.6-57.3-10.7l-1.7-0.4l0.4-1.7c5.6-18.1%2c12.6-35.9%2c20.9-52.7l0.8-1.7l1.5%2c0.8c12.6%2c7%2c26.1%2c13.5%2c39.6%2c19l1.3%2c0.4l-0.2%2c1.2c-2.2%2c14.4-3.3%2c29.1-3.7%2c43.8v1.9H93z M134.4%2c132.6v-0.4c-11.1%2c0.2-22.2%2c0.8-33.1%2c2.1c1.2%2c11.3%2c3.4%2c22.4%2c5.8%2c33.4C114.8%2c154.9%2c123.9%2c143.1%2c134.4%2c132.6L134.4%2c132.6z M94.9%2c404.4v-1.5l-1.5-0.2c-16.3-1.5-32.7-4.2-48.9-8.2l-4.4-1l2.4%2c3.8c14.7%2c22.2%2c32.6%2c41.8%2c53.4%2c58.5l3.4%2c2.8l-0.7-4.2C96.4%2c437.3%2c95.1%2c420.4%2c94.9%2c404.4zM174.2%2c14.4c-20%2c6.9-38.7%2c16-56%2c27.4c13%2c2.4%2c26%2c5.3%2c38.6%2c9.2C162%2c38.3%2c167.8%2c26.4%2c174.2%2c14.4z M256.8%2c0.7c-9.9%2c0-19.4%2c0.6-29%2c1.7c13.4%2c9.3%2c26.5%2c19.6%2c38.6%2c31l10.3%2c9.4l-10%2c9.6c-8.4%2c8-16.3%2c16.5-24%2c25.6l-0.4%2c0.4c0%2c0-1.5%2c1.8-3.8%2c4.5c6-0.7%2c12.2-1.1%2c18.3-1.1c96%2c0%2c174%2c77.8%2c174%2c174c0%2c95.9-77.8%2c174-174%2c174c-33.3%2c0-64.5-9.4-90.8-25.6c-13%2c1.5-26%2c2.1-39.3%2c2.1c-6.2%2c0-12.2-0.2-18.4-0.4c0.4%2c19.9%2c2.3%2c40.1%2c5.6%2c61.5c40.8%2c27.6%2c89.9%2c43.8%2c142.9%2c43.8C397.9%2c511.3%2c512%2c397%2c512%2c256.2C512%2c115.1%2c397.9%2c0.8%2c256.8%2c0.7L256.8%2c0.7z M223.1%2c61c5.1-5.9%2c10.3-11.8%2c15.6-17.5c-12-10.3-24.6-19.5-37.8-27.7c-8.2%2c13.6-15.2%2c27.9-21.2%2c42.5c11.5%2c4.2%2c22.6%2c9.2%2c33.8%2c14.5C218.3%2c66.2%2c222%2c62.3%2c223.1%2c61L223.1%2c61z M49.8%2c206.7l1%2c1.5l1.5-1.1c11.8-8.5%2c24.2-16.5%2c37-23.5l1.3-0.4l-0.5-1.5c-3.8-14.4-6.7-29.3-8.8-44.5l-0.2-1.7l-1.7%2c0.4c-18.8%2c3.4-37.6%2c8.5-55.5%2c14.9l-1.7%2c0.6l0.7%2c1.7C30.4%2c171.7%2c39.3%2c189.6%2c49.8%2c206.7z M46.8%2c231.4l-1.5%2c1c-13.7%2c11.2-26.7%2c23.5-38.7%2c36.8l-1.2%2c1.3l1.2%2c1.1c13.3%2c11.5%2c27.6%2c22.2%2c42.6%2c31.7l1.5%2c1.1l0.8-1.5c6.6-12%2c14.4-23.8%2c22.4-35.1l0.8-1.1l-0.8-1.1c-9.4-10.4-18.2-21.4-26.3-32.9L46.8%2c231.4z M147%2c392.2l4.5-0.2l-3.4-2.8c-13.7-11.1-25.4-24.1-35.1-39.1v-0.4l-2.8-1.3l-0.4%2c2.4c-1.7%2c12.8-2.8%2c26-3%2c39.7v1.7h1.7c5.9%2c0.4%2c12.2%2c0.4%2c18.1%2c0.4C133.4%2c392.9%2c140.1%2c392.6%2c147%2c392.2L147%2c392.2z M138.5%2c110.4c2.8-12.6%2c6.2-25%2c10.3-37.2c-16-4.5-32.5-7.9-48.9-10.3c-1.5%2c16.7-1.7%2c33.4-0.7%2c49.8C112.2%2c111.2%2c125.5%2c110.5%2c138.5%2c110.4L138.5%2c110.4z M163%2c109.5c11.2-7%2c23.1-13%2c35.7-17.5c-9-4-17.9-7.8-27.4-11.3C168.2%2c90%2c165.4%2c99.9%2c163%2c109.5z M44.5%2c316.9l-1.3-0.9c-13.2-8.3-25.9-17.6-37.8-27.3l-3.4-2.8l0.7%2c4.5c3.2%2c24.6%2c10%2c48.3%2c20%2c71l1.7%2c4.1l1.5-4.1c4.9-14.5%2c11.1-28.9%2c17.9-42.5L44.5%2c316.9z M77.1%2c74.7C62%2c89.9%2c48.5%2c106.7%2c37.6%2c125.3c13.2-3.9%2c26.5-7.1%2c39.9-9.4C76.7%2c102.2%2c76.6%2c88.6%2c77.1%2c74.7z M84.8%2c253.7l0.3-4.4c0.7-15.6%2c3.2-30.8%2c7.9-45.6l1.3-4.3l-3.9%2c2.2c-10%2c5.8-19.9%2c12.2-29.3%2c19l-1.5%2c1l1.1%2c1.5c6.7%2c9.4%2c13.7%2c18.6%2c21.2%2c27.4L84.8%2c253.7zM86.6%2c279.9l-0.7-4.2l-2.5%2c3.8c-7.1%2c10-13.7%2c20.3-19.7%2c31l-0.8%2c1.5l1.5%2c0.8c10.7%2c6%2c22%2c11.5%2c33.3%2c16.5l4.1%2c1.7l-1.9-4.1C93.2%2c311.9%2c88.7%2c296.1%2c86.6%2c279.9L86.6%2c279.9z M36.7%2c218.9l1.3-1.1l-0.8-1.3c-7.9-12.8-15.2-26.3-21.6-40l-1.9-3.8l-1.2%2c4.3c-7.5%2c23.1-11.6%2c47.3-12.2%2c71.6L0%2c253.1l3.3-3.2C13.7%2c238.8%2c25%2c228.1%2c36.7%2c218.9L36.7%2c218.9z' style='fill: green%3b'/%3e %3c/g%3e %3c/svg%3e");
  --anaconda-rs-eye-icon: url("data:image/svg+xml,%3csvg width='16' height='16' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg id='eye'%3e %3cpath id='Vector' d='M7.99861 3.11154C6.18766 3.11154 4.6989 3.93369 3.55733 4.99193C2.48798 5.98629 1.74915 7.16674 1.37141 8C1.74915 8.83326 2.48798 10.0137 3.55455 11.0081C4.6989 12.0663 6.18766 12.8885 7.99861 12.8885C9.80957 12.8885 11.2983 12.0663 12.4399 11.0081C13.5092 10.0137 14.2481 8.83326 14.6258 8C14.2481 7.16674 13.5092 5.98629 12.4427 4.99193C11.2983 3.93369 9.80957 3.11154 7.99861 3.11154ZM2.64908 4.01701C3.9573 2.80045 5.75436 1.77832 7.99861 1.77832C10.2429 1.77832 12.0399 2.80045 13.3481 4.01701C14.648 5.22524 15.5174 6.66678 15.9313 7.65837C16.0229 7.87779 16.0229 8.12221 15.9313 8.34164C15.5174 9.33322 14.648 10.7775 13.3481 11.983C12.0399 13.1996 10.2429 14.2217 7.99861 14.2217C5.75436 14.2217 3.9573 13.1996 2.64908 11.983C1.34919 10.7775 0.479819 9.33322 0.068744 8.34164C-0.0229147 8.12221 -0.0229147 7.87779 0.068744 7.65837C0.479819 6.66678 1.34919 5.22247 2.64908 4.01701ZM7.99861 10.222C9.22628 10.222 10.2206 9.22767 10.2206 8C10.2206 6.77233 9.22628 5.77797 7.99861 5.77797C7.97917 5.77797 7.9625 5.77797 7.94306 5.77797C7.97917 5.91963 7.99861 6.06961 7.99861 6.22238C7.99861 7.20285 7.20146 8 6.22099 8C6.06822 8 5.91824 7.98056 5.77658 7.94445C5.77658 7.96389 5.77658 7.98056 5.77658 8C5.77658 9.22767 6.77094 10.222 7.99861 10.222ZM7.99861 4.44476C8.94152 4.44476 9.84581 4.81932 10.5126 5.48606C11.1793 6.1528 11.5539 7.05709 11.5539 8C11.5539 8.94291 11.1793 9.8472 10.5126 10.5139C9.84581 11.1807 8.94152 11.5552 7.99861 11.5552C7.0557 11.5552 6.15141 11.1807 5.48467 10.5139C4.81793 9.8472 4.44336 8.94291 4.44336 8C4.44336 7.05709 4.81793 6.1528 5.48467 5.48606C6.15141 4.81932 7.0557 4.44476 7.99861 4.44476Z' fill='%232C303A'/%3e %3c/g%3e %3c/svg%3e");
  --anaconda-rs-settings-icon: url("data:image/svg+xml,%3csvg width='16' height='16' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg id='more-vertical'%3e %3cg id='vector'%3e %3cpath d='M7.5 4C8.325 4 9 3.325 9 2.5C9 1.675 8.325 1 7.5 1C6.675 1 6 1.675 6 2.5C6 3.325 6.675 4 7.5 4Z' fill='%238C94A6'/%3e %3cpath d='M7.5 6C6.675 6 6 6.675 6 7.5C6 8.325 6.675 9 7.5 9C8.325 9 9 8.325 9 7.5C9 6.675 8.325 6 7.5 6Z' fill='%238C94A6'/%3e %3cpath d='M7.5 11C6.675 11 6 11.675 6 12.5C6 13.325 6.675 14 7.5 14C8.325 14 9 13.325 9 12.5C9 11.675 8.325 11 7.5 11Z' fill='%238C94A6'/%3e %3c/g%3e %3c/g%3e %3c/svg%3e");
  --anaconda-rs-rocket-icon-green-icon: url("data:image/svg+xml,%3csvg width='21' height='15' viewBox='0 0 21 15' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg clip-path='url(%23clip0_360_67481)'%3e %3cpath d='M14.9261 0.369778C14.893 0.160161 14.7367 0.0566834 14.4892 0.036901C13.8292 -0.0159789 13.1714 -0.0213049 12.5139 0.083694C11.7668 0.203149 11.0453 0.411245 10.3482 0.710263C9.41091 1.112 8.55417 1.65373 7.76288 2.30883C7.71728 2.3465 7.64411 2.37997 7.59079 2.37122C7.24111 2.3134 6.89437 2.23845 6.54505 2.17834C5.76957 2.04481 5.04005 2.18253 4.39437 2.65008C4.10793 2.85741 3.8623 3.12828 3.61006 3.38279C2.9857 4.01278 1.85208 5.1769 1.74729 5.28342L0.980998 6.18504C0.83833 6.35928 0.918857 6.60998 1.17294 6.65677L2.45548 6.80704C3.05299 6.86411 3.65088 6.91661 4.24876 6.97291C4.2881 6.97672 4.32634 6.99536 4.36716 7.00791C4.24398 7.34573 4.12668 7.66644 4.01012 7.9879C3.9432 8.17241 3.98806 8.32687 4.12264 8.46382C4.37525 8.72061 4.62345 8.98159 4.87422 9.24028C5.53314 9.92125 6.19169 10.6022 6.85208 11.2817C6.96055 11.3935 7.09293 11.4312 7.24185 11.3783C7.32936 11.3475 7.4165 11.3144 7.50328 11.2805C7.7456 11.1854 7.98754 11.0896 8.24199 10.9891C8.25045 11.0519 8.25817 11.0953 8.26222 11.139C8.35414 12.1449 8.44607 13.1511 8.53726 14.157C8.5527 14.3266 8.61668 14.4609 8.77773 14.5271C8.94541 14.596 9.09028 14.5473 9.21199 14.4217L12.3856 11.0793C12.4706 10.9351 12.5617 10.8038 12.6298 10.6608C12.8912 10.1111 13.0034 9.52941 12.9206 8.91958C12.8655 8.51214 12.7816 8.10926 12.7213 7.70258C12.7107 7.6303 12.7338 7.53595 12.7743 7.47546C13.0037 7.13269 13.2574 6.80704 13.4762 6.45781C14.034 5.56798 14.4517 4.60892 14.715 3.58328C14.8889 2.90573 14.9978 2.21525 14.997 1.51069L15.0003 1.16336C14.9772 0.898576 14.9676 0.631134 14.9264 0.369397L14.9261 0.369778ZM13.5472 4.63593C12.9438 6.07129 12.0514 7.28449 10.9167 8.31431C10.4033 8.77996 9.84995 9.18588 9.26641 9.54957C9.21052 9.58419 9.15279 9.61538 9.05571 9.67093C8.8134 9.40767 8.57108 9.1349 8.31884 8.87202C7.47534 7.99399 6.63036 7.11748 5.78023 6.24629C5.68463 6.14814 5.67544 6.07928 5.74236 5.96287C6.47923 4.685 7.404 3.58137 8.55454 2.68318C9.21934 2.16465 9.93893 1.74922 10.7038 1.41216C10.8608 1.34292 10.9582 1.357 11.0836 1.489C11.8499 2.2959 12.6246 3.09404 13.4016 3.8899C13.7207 4.21707 13.7255 4.21061 13.5468 4.63555L13.5472 4.63593Z' fill='%230CCA4A'/%3e %3cpath d='M9.93641 3.83131C9.18004 3.82941 8.57996 4.45065 8.57996 5.23586C8.57996 6.03172 9.1635 6.63965 9.92795 6.64041C10.6891 6.64155 11.2877 6.02944 11.2936 5.24385C11.2991 4.47272 10.685 3.83322 9.93641 3.83093V3.83131Z' fill='%230CCA4A'/%3e %3cpath d='M0.611532 14.8558C0.478057 14.8558 0.344949 14.7999 0.247876 14.6899C0.0647606 14.4822 0.0787332 14.16 0.279866 13.9706L4.09182 10.3721C4.29259 10.1826 4.60403 10.1974 4.78715 10.4052C4.97026 10.6129 4.95629 10.9351 4.75516 11.1246L0.943199 14.7227C0.848699 14.8117 0.729932 14.8558 0.611532 14.8558Z' fill='%230CCA4A'/%3e %3cpath d='M3.4432 15C3.30972 15 3.17661 14.9441 3.07954 14.8341C2.89643 14.6264 2.9104 14.3042 3.11153 14.1147L4.63087 12.6805C4.83164 12.4911 5.14308 12.5059 5.3262 12.7136C5.50931 12.9213 5.49534 13.2436 5.2942 13.433L3.77486 14.8672C3.68036 14.9563 3.5616 15.0004 3.4432 15.0004V15Z' fill='%230CCA4A'/%3e %3cpath d='M0.492392 12.4005C0.358916 12.4005 0.225808 12.3446 0.128735 12.2346C-0.05438 12.0269 -0.0404074 11.7047 0.160725 11.5153L1.68007 10.081C1.88083 9.89157 2.19191 9.90641 2.37539 10.1141C2.5585 10.3218 2.54453 10.6441 2.3434 10.8335L0.824058 12.2677C0.729559 12.3568 0.610791 12.4009 0.492392 12.4009V12.4005Z' fill='%230CCA4A'/%3e %3c/g%3e %3cpath d='M17.5 6.5V4.5H16.5V6.5H14.5V7.5H16.5V9.5H17.5V7.5H19.5V6.5H17.5Z' fill='%230CCA4A'/%3e %3cdefs%3e %3cclipPath id='clip0_360_67481'%3e %3crect width='15' height='15' fill='white'/%3e %3c/clipPath%3e %3c/defs%3e %3c/svg%3e");
  --anaconda-rs-rocket-icon-gray-icon: url("data:image/svg+xml,%3csvg width='18' height='18' viewBox='0 0 18 18' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg clip-path='url(%23clip0_345_19040)'%3e %3cpath d='M17.9113 0.428824C17.8716 0.1857 17.6841 0.0656821 17.3871 0.0427376C16.5951 -0.018595 15.8057 -0.0247723 15.0168 0.0970103C14.1202 0.23556 13.2544 0.476919 12.4179 0.823735C11.2931 1.28969 10.265 1.91801 9.31549 2.67783C9.26077 2.72151 9.17297 2.76034 9.10899 2.75019C8.68937 2.68313 8.27328 2.5962 7.8541 2.52648C6.92352 2.37161 6.0481 2.53134 5.27328 3.07362C4.92955 3.3141 4.6348 3.62826 4.33211 3.92345C3.58288 4.65415 2.22253 6.00435 2.09678 6.1279L1.17723 7.17364C1.00603 7.37573 1.10266 7.66651 1.40756 7.72078L2.94661 7.89507C3.66363 7.96125 4.38109 8.02215 5.09855 8.08745C5.14576 8.09186 5.19165 8.11348 5.24062 8.12804C5.09281 8.51987 4.95205 8.89183 4.81218 9.26468C4.73187 9.47868 4.7857 9.65783 4.9472 9.81667C5.25033 10.1145 5.54817 10.4172 5.8491 10.7172C6.6398 11.5071 7.43006 12.2969 8.22253 13.0849C8.3527 13.2147 8.51155 13.2584 8.69025 13.197C8.79526 13.1613 8.89984 13.1229 9.00397 13.0836C9.29475 12.9733 9.58509 12.8621 9.89043 12.7456C9.90057 12.8184 9.90984 12.8687 9.91469 12.9195C10.025 14.0861 10.1353 15.2532 10.2447 16.4199C10.2633 16.6166 10.3401 16.7724 10.5333 16.8492C10.7345 16.929 10.9084 16.8726 11.0544 16.727L14.8628 12.8502C14.9647 12.683 15.0741 12.5308 15.1558 12.3648C15.4695 11.7272 15.6041 11.0526 15.5048 10.3453C15.4386 9.87271 15.338 9.40544 15.2656 8.93375C15.2528 8.84991 15.2806 8.74049 15.3292 8.67033C15.6045 8.27277 15.909 7.89507 16.1715 7.49001C16.8409 6.45795 17.3421 5.34558 17.658 4.15599C17.8667 3.37014 17.9974 2.56929 17.9965 1.75211L18.0004 1.34925C17.9726 1.04215 17.9612 0.731957 17.9118 0.428383L17.9113 0.428824ZM16.2567 5.3769C15.5326 7.04171 14.4617 8.44883 13.1 9.64327C12.484 10.1833 11.82 10.6542 11.1197 11.076C11.0527 11.1161 10.9834 11.1523 10.8669 11.2167C10.5761 10.9114 10.2853 10.595 9.98265 10.2901C8.97044 9.27174 7.95646 8.25512 6.93631 7.24468C6.82159 7.13084 6.81056 7.05097 6.89087 6.91595C7.77511 5.43382 8.88484 4.15378 10.2655 3.11201C11.0632 2.5106 11.9268 2.02876 12.8445 1.63783C13.0329 1.55752 13.1499 1.57385 13.3003 1.72696C14.2199 2.66283 15.1496 3.58855 16.0819 4.51163C16.4649 4.8911 16.4707 4.8836 16.2562 5.37646L16.2567 5.3769Z' fill='%23616161'/%3e %3cpath d='M11.9237 4.4438C11.016 4.44159 10.2959 5.16214 10.2959 6.07286C10.2959 6.99594 10.9962 7.70104 11.9135 7.70193C12.8269 7.70325 13.5452 6.99329 13.5523 6.08213C13.5589 5.18773 12.822 4.44601 11.9237 4.44336V4.4438Z' fill='%23616161'/%3e %3cpath d='M0.733782 17.2304C0.573611 17.2304 0.413882 17.1655 0.297394 17.038C0.0776562 16.7971 0.0944234 16.4234 0.335782 16.2036L4.91013 12.0299C5.15105 11.8102 5.52478 11.8274 5.74452 12.0683C5.96426 12.3092 5.94749 12.683 5.70613 12.9027L1.13178 17.0759C1.01838 17.1792 0.875862 17.2304 0.733782 17.2304Z' fill='%23616161'/%3e %3cpath d='M4.13178 17.3977C3.97161 17.3977 3.81188 17.3328 3.6954 17.2053C3.47566 16.9644 3.49243 16.5906 3.73379 16.3709L5.55699 14.7074C5.79791 14.4877 6.17164 14.5049 6.39138 14.7458C6.61112 14.9867 6.59435 15.3604 6.35299 15.5802L4.52978 17.2437C4.41639 17.3469 4.27386 17.3981 4.13178 17.3981V17.3977Z' fill='%23616161'/%3e %3cpath d='M0.590823 14.3828C0.430652 14.3828 0.270923 14.3179 0.154435 14.1904C-0.0653034 13.9495 -0.0485362 13.5757 0.192823 13.356L2.01603 11.6925C2.25695 11.4728 2.63024 11.49 2.85042 11.7309C3.07016 11.9718 3.05339 12.3455 2.81203 12.5653L0.988822 14.2288C0.875423 14.332 0.732902 14.3832 0.590823 14.3832V14.3828Z' fill='%23616161'/%3e %3c/g%3e %3cdefs%3e %3cclipPath id='clip0_345_19040'%3e %3crect width='18' height='17.3977' fill='white'/%3e %3c/clipPath%3e %3c/defs%3e %3c/svg%3e");
  --anaconda-rs-log-icon: url("data:image/svg+xml,%3csvg width='16' height='16' viewBox='0 0 16 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath d='M9.44667 2H3.33333C2.6 2 2 2.6 2 3.33333V12.6667C2 13.4 2.6 14 3.33333 14H12.6667C13.4 14 14 13.4 14 12.6667V6.55333C14 6.2 13.86 5.86 13.6067 5.61333L10.3867 2.39333C10.14 2.14 9.8 2 9.44667 2ZM5.33333 10H10.6667C11.0333 10 11.3333 10.3 11.3333 10.6667C11.3333 11.0333 11.0333 11.3333 10.6667 11.3333H5.33333C4.96667 11.3333 4.66667 11.0333 4.66667 10.6667C4.66667 10.3 4.96667 10 5.33333 10ZM5.33333 7.33333H10.6667C11.0333 7.33333 11.3333 7.63333 11.3333 8C11.3333 8.36667 11.0333 8.66667 10.6667 8.66667H5.33333C4.96667 8.66667 4.66667 8.36667 4.66667 8C4.66667 7.63333 4.96667 7.33333 5.33333 7.33333ZM5.33333 4.66667H8.66667C9.03333 4.66667 9.33333 4.96667 9.33333 5.33333C9.33333 5.7 9.03333 6 8.66667 6H5.33333C4.96667 6 4.66667 5.7 4.66667 5.33333C4.66667 4.96667 4.96667 4.66667 5.33333 4.66667Z' fill='%232C303A'/%3e %3c/svg%3e");
  --anaconda-primary-color: #43b049;
  --anaconda-apps-alert-color: #d50000;
  --anaconda-apps-download-icon: url("data:image/svg+xml,%3csvg width='36' height='36' viewBox='0 0 36 36' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath fill-rule='evenodd' clip-rule='evenodd' d='M16.75 9.25H19.25V14.25H23L18 20.5L13 14.25H16.75V9.25ZM9.25 26.75V23H26.75V26.75H9.25Z' fill='%23CED2D9'/%3e %3c/svg%3e");
  --anaconda-apps-delete-icon: url("data:image/svg+xml,%3csvg width='36' height='36' viewBox='0 0 36 36' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg clip-path='url(%23clip0_996_9916)'%3e %3cpath d='M12.9998 23.8333C12.9998 24.75 13.7498 25.5 14.6665 25.5H21.3332C22.2498 25.5 22.9998 24.75 22.9998 23.8333V15.5C22.9998 14.5833 22.2498 13.8333 21.3332 13.8333H14.6665C13.7498 13.8333 12.9998 14.5833 12.9998 15.5V23.8333ZM15.4998 15.5H20.4998C20.9582 15.5 21.3332 15.875 21.3332 16.3333V23C21.3332 23.4583 20.9582 23.8333 20.4998 23.8333H15.4998C15.0415 23.8333 14.6665 23.4583 14.6665 23V16.3333C14.6665 15.875 15.0415 15.5 15.4998 15.5ZM20.9165 11.3333L20.3248 10.7417C20.1748 10.5917 19.9582 10.5 19.7415 10.5H16.2582C16.0415 10.5 15.8248 10.5917 15.6748 10.7417L15.0832 11.3333H12.9998C12.5415 11.3333 12.1665 11.7083 12.1665 12.1667C12.1665 12.625 12.5415 13 12.9998 13H22.9998C23.4582 13 23.8332 12.625 23.8332 12.1667C23.8332 11.7083 23.4582 11.3333 22.9998 11.3333H20.9165Z' fill='%23CED2D9'/%3e %3c/g%3e %3cdefs%3e %3cclipPath id='clip0_996_9916'%3e %3crect width='20' height='20' fill='white' transform='translate(8 8)'/%3e %3c/clipPath%3e %3c/defs%3e %3c/svg%3e");
  --anaconda-apps-preview-icon: url("data:image/svg+xml,%3csvg width='21' height='21' viewBox='0 0 21 21' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg id='preview'%3e %3cpath id='vector' fill-rule='evenodd' clip-rule='evenodd' d='M1.75 10.5C3.47656 7.51172 6.79688 5.5 10.5 5.5C14.2031 5.5 17.5234 7.51172 19.25 10.5C17.5234 13.4883 14.1992 15.5 10.5 15.5C6.80078 15.5 3.48047 13.4883 1.75 10.5ZM10.5 9.25C10.5 8.55859 9.94141 8 9.25 8C8.55859 8 8 8.55859 8 9.25C8 9.94141 8.55859 10.5 9.25 10.5C9.94141 10.5 10.5 9.94141 10.5 9.25ZM3.22659 10.5C3.94534 9.46875 4.97659 8.59766 6.06253 7.95703C6.53909 7.67578 7.043 7.44141 7.56253 7.25781C6.67972 8.05859 6.12503 9.21484 6.12503 10.5C6.12503 11.7852 6.67972 12.9414 7.56253 13.7422C7.043 13.5586 6.53909 13.3242 6.06253 13.043C4.97659 12.4023 3.94534 11.5312 3.22659 10.5ZM13.4375 13.7422C13.957 13.5586 14.4609 13.3242 14.9375 13.043C16.0234 12.4023 17.0508 11.5312 17.7734 10.5C17.0547 9.46875 16.0234 8.59766 14.9375 7.95703C14.4609 7.67578 13.957 7.44141 13.4375 7.25781C14.3203 8.05859 14.875 9.21484 14.875 10.5C14.875 11.7852 14.3203 12.9414 13.4375 13.7422Z' fill='%232C303A'/%3e %3c/g%3e %3c/svg%3e");
  --anaconda-apps-notebook-icon: url("data:image/svg+xml,%3csvg width='19' height='21' viewBox='0 0 19 21' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg id='Layer_1-2'%3e %3cpath id='Vector' d='M15.45 1.5C16.416 1.5 17.2 2.24667 17.2 3.16667V17.8333C17.2 18.7533 16.416 19.5 15.45 19.5H3.55C2.584 19.5 1.8 18.7533 1.8 17.8333V3.16667C1.8 2.24667 2.584 1.5 3.55 1.5H15.45ZM15.45 0.5H3.55C2.01 0.5 0.75 1.7 0.75 3.16667V17.8333C0.75 19.3 2.01 20.5 3.55 20.5H15.45C16.99 20.5 18.25 19.3 18.25 17.8333V3.16667C18.25 1.7 16.99 0.5 15.45 0.5Z' fill='%232C303A'/%3e %3cpath id='Vector_2' d='M11.3831 7.23333L12.1251 6.52666L14.5401 8.82333L12.1251 11.1233L11.3831 10.4167L13.0526 8.82333L11.3831 7.23333Z' fill='%232C303A'/%3e %3cpath id='Vector_3' d='M9.72464 6.13638L8.28003 11.0508L9.29115 11.3204L10.7358 6.40597L9.72464 6.13638Z' fill='%232C303A'/%3e %3cpath id='Vector_4' d='M6.19258 3.92C6.53085 3.92 6.80508 3.65883 6.80508 3.33667C6.80508 3.0145 6.53085 2.75333 6.19258 2.75333C5.8543 2.75333 5.58008 3.0145 5.58008 3.33667C5.58008 3.65883 5.8543 3.92 6.19258 3.92Z' fill='%232C303A'/%3e %3cpath id='Vector_5' d='M6.76998 6.52666L7.51198 7.23333L5.83898 8.82333L7.51198 10.4167L6.76998 11.1233L4.35498 8.82333L6.76998 6.52666Z' fill='%232C303A'/%3e %3cpath id='Vector_6' d='M4.09248 3.92C4.43075 3.92 4.70498 3.65883 4.70498 3.33667C4.70498 3.0145 4.43075 2.75333 4.09248 2.75333C3.75421 2.75333 3.47998 3.0145 3.47998 3.33667C3.47998 3.65883 3.75421 3.92 4.09248 3.92Z' fill='%232C303A'/%3e %3cpath id='Vector_7' d='M9.25503 16.92H3.65503V17.5867H9.25503V16.92Z' fill='%232C303A'/%3e %3cpath id='Vector_8' d='M15.205 14.92H3.65503V15.5867H15.205V14.92Z' fill='%232C303A'/%3e %3c/g%3e %3c/svg%3e");
  --anaconda-apps-check-icon: url("data:image/svg+xml,%3csvg width='17' height='16' viewBox='0 0 17 16' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg id='check-outline'%3e %3cpath id='vector' fill-rule='evenodd' clip-rule='evenodd' d='M8.5 1C4.63437 1 1.5 4.13437 1.5 8C1.5 11.8656 4.63437 15 8.5 15C12.3656 15 15.5 11.8656 15.5 8C15.5 4.13437 12.3656 1 8.5 1ZM8.5 14C5.1875 14 2.5 11.3125 2.5 8C2.5 4.6875 5.1875 2 8.5 2C11.8125 2 14.5 4.6875 14.5 8C14.5 11.3125 11.8125 14 8.5 14ZM4.90625 6.59375L7.5 9.1875L11.5938 5.09375L13 6.5L7.5 12L3.5 8L4.90625 6.59375Z' fill='%230CCA4A'/%3e %3c/g%3e %3c/svg%3e");
  --anaconda-apps-tile-border: var(--jp-border-color3);
  --anaconda-apps-tile-background: var(--jp-layout-color0);
  --anaconda-apps-tile-background-hover: var(--jp-layout-color2);
}
.anaconda-app-status-publishing-icon {
  -webkit-mask-image: var(--anaconda-rs-running-icon);
  mask-image: var(--anaconda-rs-running-icon);
  background: #ffba0a;
  width: 16px;
  height: 12px;
}

.anaconda-app-publish-green-icon {
  -webkit-mask-image: var(--anaconda-rs-rocket-icon-green-icon);
  mask-image: var(--anaconda-rs-rocket-icon-green-icon);
  width: 20px;
  height: 15px;
  flex-shrink: 0;
  background: #0CCA4A;
  margin-left: 4px;
  margin-right: 4px;
}

.anaconda-app-publish-gray-icon {
  -webkit-mask-image: var(--anaconda-rs-rocket-icon-gray-icon);
  mask-image: var(--anaconda-rs-rocket-icon-gray-icon);
  width: 18px;
  height: 18px;
  flex-shrink: 0;
  background: #2C303A;
  margin-right: 4px;
}

.anaconda-apps-download-icon {
  -webkit-mask-image: var(--anaconda-apps-download-icon);
  mask-image: var(--anaconda-apps-download-icon);
  width: 30px;
  height: 30px;
  background: #2C303A;
}

.anaconda-apps-delete-icon {
  -webkit-mask-image: var(--anaconda-apps-delete-icon);
  mask-image: var(--anaconda-apps-delete-icon);
  width: 30px;
  height: 30px;
  background: #2C303A;
}

.anaconda-app-log-icon {
  -webkit-mask-image: var(--anaconda-rs-log-icon);
  mask-image: var(--anaconda-rs-log-icon);
  width: 18px;
  height: 18px;
  flex-shrink: 0;
  background: #2C303A;
  margin-right: 4px;
}

.anaconda-app-status-published-icon {
  -webkit-mask-image: var(--anaconda-rs-check-icon);
  mask-image: var(--anaconda-rs-check-icon);
  width: 16px;
  height: 16px;
  background: #0cca4a;
}

.anaconda-app-status-failed-icon {
  -webkit-mask-image: var(--anaconda-rs-warning-icon);
  mask-image: var(--anaconda-rs-warning-icon);
  width: 16px;
  height: 16px;
  background: #DC3246;
}

.anaconda-app-eye-icon {
  -webkit-mask-image: var(--anaconda-rs-eye-icon);
  mask-image: var(--anaconda-rs-eye-icon);
  width: 16px;
  height: 16px;
  background: var(--jp-ui-font-color0);
  margin-right: 4px;
  align-self: center;
  cursor: pointer;
}

.anaconda-app-settings-icon {
  -webkit-mask-image: var(--anaconda-rs-settings-icon);
  mask-image: var(--anaconda-rs-settings-icon);
  width: 16px;
  height: 16px;
  background: #2C303A;
  margin-left: auto;
}

.anaconda-app-settings-gray-icon {
  -webkit-mask-image: var(--anaconda-rs-settings-icon);
  mask-image: var(--anaconda-rs-settings-icon);
  width: 16px;
  height: 16px;
  background: #6A718A;
  cursor: pointer;
}

.anaconda-app-list {
	display: flex;
  gap: 25px;
  flex-wrap: wrap;
  margin-bottom: 40px;
}

.anaconda-app-div {
  margin-top: 25px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.anaconda-app-header {
    display: inline-flex;
    flex: 0 0 auto;
    position: relative;
    gap: 10px;
    justify-content: space-between;
}

.anaconda-app-header h3 {
  font-weight: 500;
  font-size: 34px;
  line-height: 38px;
  margin: 0;
  margin-bottom: 25px;
}

.anaconda-app-tile {
	width: 260px;
    border: 1px solid;
    border-radius: 8px;
    background-color: var(--anaconda-lms-tile-background);
    border: 1px solid var(--gray-400, #8C94A6);
    cursor: pointer;
    display: flex;
}

.anaconda-app-frame {
	align-items: flex-start;
    display: flex;
    position: relative;
    width: -webkit-fill-available;
}

.anaconda-app-gradient-published,
.anaconda-app-gradient-publishing,
.anaconda-app-gradient-failed,
.anaconda-app-gradient-inactive {
  border-radius: 6px 0px 0px 6px;
  height: 107px;
  position: relative;
  width: 6px;
}

.anaconda-app-gradient-published {
  background: #0CCA4A;
}

.anaconda-app-gradient-publishing {
  background: #FFC105;
}

.anaconda-app-gradient-failed {
  background: #DC3246;
}

.anaconda-app-gradient-inactive {
  background: #8C94A6;
}

.anaconda-app-container {
	display: flex;
    flex-direction: column;
    height: 67px;
    justify-content: space-between;
    margin-right: -10px;
    padding: 12px;
    position: relative;
    gap: 6px;
    width: -webkit-fill-available;
}

.anaconda-app-header-div, .anaconda-app-status-div {
	align-items: center;
    display: inline-flex;
    flex: 0 0 auto;
    position: relative;
    width: 225px;
    gap: 8px;
}

.anaconda-app-header-div {
	justify-content: flex-start;
}

.anaconda-app-status-div {
	justify-content: space-between;
}

.anaconda-app-subheader {
	display: flex;
    margin: 10px 0px;
}

.anaconda-app-subheader a {
  color: #43b049;
  font-weight: 700;
  text-decoration: underline;
  cursor: pointer;
}

.anaconda-app-header-text {
	  color: var(--anaconda-lms-default-color);
    font-family: var(--anaconda-lms-font-family);
    font-size: 16px;
    font-weight: 700;
    text-transform: capitalize;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    width: 165px;
}

.anaconda-app-launcher {
  display: flex;
  flex-direction: column;
  padding: 42px;
  height: calc(100% - 60px);
  overflow: auto;
}

.anaconda-app-version-published,
.anaconda-app-version-publishing,
.anaconda-app-version-inactive,
.anaconda-app-version-failed
 {
	background: rgba(67, 176, 73, 0.12);
    border-radius: 8px;
    padding: 5px;
    margin-top: 10px;
    width: max-content;
}

.anaconda-app-version-publishing {
	background: rgba(255, 193, 5, 0.12);
}

.anaconda-app-version-inactive {
	background: rgba(44, 48, 58, 0.12)
}

.anaconda-app-version-failed {
	background: rgba(220, 50, 70, 0.12);
}

.anaconda-app-status {
	display: flex;
    align-self: end;
}

.anaconda-app-status span {
	font-size: 12px;
    font-style: normal;
    font-weight: 700;
}

/* Apps Error Container Styles */
.anaconda-apps-error {
  background: var(--anaconda-apps-tile-background);
  border: 0.5px solid var(--anaconda-apps-tile-border);
  border-radius: 4px;
  position: relative;
  padding: 17px 36px;
  display: flex;
  flex-direction: column;
  width: -webkit-fill-available;
}

.anaconda-apps-tile-left-border {
  position: absolute;
  width: 5px;
  height: calc(100% + 1px);
  background: var(--anaconda-apps-alert-color);
  border-radius: 4px 0 0 4px;
  top: -1px;
  left: -1px;
}

.anaconda-apps-modal .jp-Dialog-content {
  border-radius: 8px;
  background: var(--bg-color, #fff);
  box-shadow: 0px 4px 4px 0px rgba(0, 0, 0, 0.25);
  backdrop-filter: blur(5px);
  width: 520px;
  min-height: 480px;
  max-height: 520px;
  padding: 0px;
}

.anaconda-apps-modal .jp-Dialog-header, .anaconda-versions-modal .jp-Dialog-header {
    border-bottom: 1px solid var(--gray-200, #ced2d9);
    padding: 15px 20px;
    background: var(--jp-layout-color1);
    font-size: 22px;
    font-style: normal;
    font-weight: 700;
}

.anaconda-apps-modal .jp-Dialog-footer {
  margin: 0px;
  padding: 0px;
}

.anaconda-apps-modal .jp-Dialog-body, .anaconda-versions-modal .jp-Dialog-body  {
  overflow: hidden !important;
}

#topbar-app-counter {
	display: flex;
}

.anaconda-versions-modal .jp-Dialog-content {
  border-radius: 8px;
  background: var(--bg-color, #fff);
  box-shadow: 0px 4px 4px 0px rgba(0, 0, 0, 0.25);
  backdrop-filter: blur(5px);
  width: 806px;
  min-height: 450px;
  padding: 0px;
}

.anaconda-versions-modal .jp-Dialog-header {
  padding: 15px 20px;
  background: var(--jp-layout-color1);
}

.anaconda-versions-modal .jp-Dialog-footer {
  margin: 0px;
  padding: 0px;
}

.anaconda-versions-modal__grid-container {
  width: 470px;
  height: 470px;
}

.paDeploy svg {
	width: 24px !important;
}

.anaconda-panel-tabs {
  font-family:arial;
  margin:0 auto;
  display: flex;
  text-align:center;
  padding: 20px;
  gap: 24px;
}

.anaconda-panel-tab-active, .anaconda-panel-tab-inactive {
  font-size: 16px;
  font-style: normal;
  font-weight: 700;
  cursor: pointer;
}

.anaconda-panel-tab-inactive {
	color: var(--gray-400, #8C94A6);
}
.anaconda-panel-tab-active {
  border-bottom: 2px solid var(--jp-ui-font-color1);
  padding-bottom: 5px;
  color: var(--jp-ui-font-color1);
}

.anaconda-apps-preview-icon {
  -webkit-mask-image: var(--anaconda-apps-preview-icon);
  mask-image: var(--anaconda-apps-preview-icon);
  width: 20px;
  height: 20px;
  background: #2C303A;
  cursor: pointer;
}

.anaconda-apps-notebook-icon {
  -webkit-mask-image: var(--anaconda-apps-notebook-icon);
  mask-image: var(--anaconda-apps-notebook-icon);
  width: 19px;
  height: 20px;
  background: var(--jp-ui-font-color0);
  cursor: pointer;
  align-self: end;
}

.anaconda-apps-check-icon {
  -webkit-mask-image: var(--anaconda-apps-check-icon);
  mask-image: var(--anaconda-apps-check-icon);
  width: 16px;
  height: 16px;
  background: #0CCA4A;
  margin-right: 8px;
}

.anaconda-versions-modal__grid-container {
  width: auto;
  padding: 0px 20px;
}

.anaconda-versions-modal .anaconda-versions-modal__grid-container {
	padding: 20px !important;
}

.anaconda-versions-modal .anaconda-versions-modal__grid-container .ag-root-wrapper {
	height: 400px !important;
}

.anaconda-versions-modal__grid-container .ag-root-wrapper {
    height: 380px !important;
    overflow: auto;
}

.anaconda-versions-modal__grid-container .ag-cell {
	display: flex;
	align-items: center;
}

.anaconda-apps-grid-action-button-cell {
  display: flex;
  align-items: center;
  height: 100%;
  gap: 12px;
}

.anaconda-apps-grid-action-background {
	border-radius: 100px;
	background: var(--buttons-secondary-secondary-button-bg, #E5E8EB);
	padding: 10px;
}

input[name="published"] {
	accent-color: rgb(77 162 97);
    width: 20px;
    height: 20px;
    transform: scale(0.9);

}

.anaconda-app-list-desc {
	font-size: 16px;
    font-weight: 400;
}

.anaconda-example-app-tile {
    width: 260px;
    height: 200px;
    border-radius: 8px;
    background-color: var(--anaconda-lms-tile-background);
    border: 1px solid var(--gray-400, #8C94A6);
    cursor: pointer;
    display: flex;
}

.anaconda-example-app-frame {
    position: relative;
    width: 260px;
    align-items: flex-start;
    display: flex;
}

.anaconda-example-app-container {
	padding: 12px;
    height: -webkit-fill-available;
}

.anaconda-example-app-gradient {
	border-radius: 6px 0px 0px 6px;
	height: 200px;
    width: 6px;
	position: relative;
	background: #0CCA4A;
}

.anaconda-example-app-preview {
	width: 230px;
    height: 120px;
    border-radius: 4px;
    background-position: center top;
	background-repeat: no-repeat;
	background-size: cover;
}

.anaconda-example-app-title {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 10px;
}
.anaconda-example-app-owner {
	width: 25px;
    height: 20px;
    border-radius: 90px;
    background-color: #43b049;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    margin-top: 10px;
    cursor: pointer;
}

.anaconda-example-app-action-div {
	align-items: center;
    display: inline-flex;
    flex: 0 0 auto;
    position: relative;
    width: 230px;
    gap: 8px;
}

.anaconda-example-app-actions {
	display: flex;
    align-self: end;
    justify-content: flex-end;
    width: -webkit-fill-available;
    gap: 10px;
}

.anaconda-app-resources-header {
	display: flex;
    width: -webkit-fill-available;
    flex-direction: row;
}

.anaconda-app-resources-collapsible-div {
	align-self: center;
    margin-right: 30px;
}

.anaconda-app-resources-collapsible {
	display: flex;
    gap: 10px;
    font-size: 16px;
    color: #6C5AF6;
    cursor: pointer;
}

.anaconda-app-resources-collapsible span {
	align-self: end;
}
</style><style>/*
    See the JupyterLab Developer Guide for useful CSS Patterns:

    https://jupyterlab.readthedocs.io/en/stable/developer/css.html
*/
</style><style>/*
    See the JupyterLab Developer Guide for useful CSS Patterns:

    https://jupyterlab.readthedocs.io/en/stable/developer/css.html
*/

:root {
  --anaconda-catalog-default-color: var(--jp-ui-font-color1);
  --anaconda-catalog-primary-color: #43b049;
  --anaconda-catalog-primary-color-hover: #337234;
  --anaconda-catalog-alert-color: #d50000;
  --anaconda-catalog-tile-background: var(--jp-layout-color1);
  /* #fcfcfc; */
  --anaconda-catalog-tile-border: var(--jp-border-color3);
  --anaconda-catalog-tile-shadow: drop-shadow(2px 2px 4px var(--jp-layout-color2));
  --anaconda-catalog-tile-toggle-background: var(--jp-layout-color1);
  --anaconda-catalog-tile-toggle-background-hover: var(--jp-layout-color2);
  --anaconda-catalog-open-notebook-border: #0cca4a;
  --anaconda-catalog-open-notebook-background: #0cca4a;
  --anaconda-catalog-open-notebook-hover: #4fab4f;
  --anaconda-catalog-default-icon-image: url("data:image/svg+xml,%3csvg width='30' height='31' viewBox='0 0 30 31' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath d='M30 4.8125C30 4.82031 30 4.82812 30 4.83594C29.9609 7.48438 23.2578 9.625 15 9.625C6.74219 9.625 0.0390625 7.48438 0 4.83594C0 4.82812 0 4.82031 0 4.8125C0 2.15625 6.71875 0 15 0C23.2812 0 30 2.15625 30 4.8125ZM30 8.4375C30 8.4375 30 8.45312 30 8.46094C29.9609 11.1094 23.2578 13.25 15 13.25C6.74219 13.25 0.0390625 11.1094 0 8.46094V8.4375L0.078125 15.2187C0.078125 15.2187 0.078125 15.25 0.078125 15.2578C0.117188 17.8906 6.78125 20.0234 15 20.0234C23.2188 20.0234 29.8828 17.8906 29.9219 15.2578C29.9219 15.25 29.9219 15.2187 29.9219 15.2187L30 8.4375ZM30 18.4375C30 18.4375 30 18.4531 30 18.4609C29.9609 21.1094 23.2578 23.25 15 23.25C6.74219 23.25 0.0390625 21.1094 0 18.4609V18.4375L0.078125 25.2188C0.078125 25.2188 0.078125 25.25 0.078125 25.2578C0.117188 27.8906 6.78125 30.0234 15 30.0234C23.2188 30.0234 29.8828 27.8906 29.9219 25.2578C29.9219 25.25 29.9219 25.2188 29.9219 25.2188L30 18.4375Z' fill='%2343B049'/%3e %3c/svg%3e");
  --anaconda-catalog-icon-image: url("data:image/svg+xml,%3csvg version='1.1' id='Layer_1' xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' x='0px' y='0px' viewBox='0 0 512 512' style='enable-background:new 0 0 512 512%3b' xml:space='preserve'%3e %3cg%3e %3cpath d='M93%2c391.4c-19.3-1.9-38.5-5.6-57.3-10.7l-1.7-0.4l0.4-1.7c5.6-18.1%2c12.6-35.9%2c20.9-52.7l0.8-1.7l1.5%2c0.8c12.6%2c7%2c26.1%2c13.5%2c39.6%2c19l1.3%2c0.4l-0.2%2c1.2c-2.2%2c14.4-3.3%2c29.1-3.7%2c43.8v1.9H93z M134.4%2c132.6v-0.4c-11.1%2c0.2-22.2%2c0.8-33.1%2c2.1c1.2%2c11.3%2c3.4%2c22.4%2c5.8%2c33.4C114.8%2c154.9%2c123.9%2c143.1%2c134.4%2c132.6L134.4%2c132.6z M94.9%2c404.4v-1.5l-1.5-0.2c-16.3-1.5-32.7-4.2-48.9-8.2l-4.4-1l2.4%2c3.8c14.7%2c22.2%2c32.6%2c41.8%2c53.4%2c58.5l3.4%2c2.8l-0.7-4.2C96.4%2c437.3%2c95.1%2c420.4%2c94.9%2c404.4zM174.2%2c14.4c-20%2c6.9-38.7%2c16-56%2c27.4c13%2c2.4%2c26%2c5.3%2c38.6%2c9.2C162%2c38.3%2c167.8%2c26.4%2c174.2%2c14.4z M256.8%2c0.7c-9.9%2c0-19.4%2c0.6-29%2c1.7c13.4%2c9.3%2c26.5%2c19.6%2c38.6%2c31l10.3%2c9.4l-10%2c9.6c-8.4%2c8-16.3%2c16.5-24%2c25.6l-0.4%2c0.4c0%2c0-1.5%2c1.8-3.8%2c4.5c6-0.7%2c12.2-1.1%2c18.3-1.1c96%2c0%2c174%2c77.8%2c174%2c174c0%2c95.9-77.8%2c174-174%2c174c-33.3%2c0-64.5-9.4-90.8-25.6c-13%2c1.5-26%2c2.1-39.3%2c2.1c-6.2%2c0-12.2-0.2-18.4-0.4c0.4%2c19.9%2c2.3%2c40.1%2c5.6%2c61.5c40.8%2c27.6%2c89.9%2c43.8%2c142.9%2c43.8C397.9%2c511.3%2c512%2c397%2c512%2c256.2C512%2c115.1%2c397.9%2c0.8%2c256.8%2c0.7L256.8%2c0.7z M223.1%2c61c5.1-5.9%2c10.3-11.8%2c15.6-17.5c-12-10.3-24.6-19.5-37.8-27.7c-8.2%2c13.6-15.2%2c27.9-21.2%2c42.5c11.5%2c4.2%2c22.6%2c9.2%2c33.8%2c14.5C218.3%2c66.2%2c222%2c62.3%2c223.1%2c61L223.1%2c61z M49.8%2c206.7l1%2c1.5l1.5-1.1c11.8-8.5%2c24.2-16.5%2c37-23.5l1.3-0.4l-0.5-1.5c-3.8-14.4-6.7-29.3-8.8-44.5l-0.2-1.7l-1.7%2c0.4c-18.8%2c3.4-37.6%2c8.5-55.5%2c14.9l-1.7%2c0.6l0.7%2c1.7C30.4%2c171.7%2c39.3%2c189.6%2c49.8%2c206.7z M46.8%2c231.4l-1.5%2c1c-13.7%2c11.2-26.7%2c23.5-38.7%2c36.8l-1.2%2c1.3l1.2%2c1.1c13.3%2c11.5%2c27.6%2c22.2%2c42.6%2c31.7l1.5%2c1.1l0.8-1.5c6.6-12%2c14.4-23.8%2c22.4-35.1l0.8-1.1l-0.8-1.1c-9.4-10.4-18.2-21.4-26.3-32.9L46.8%2c231.4z M147%2c392.2l4.5-0.2l-3.4-2.8c-13.7-11.1-25.4-24.1-35.1-39.1v-0.4l-2.8-1.3l-0.4%2c2.4c-1.7%2c12.8-2.8%2c26-3%2c39.7v1.7h1.7c5.9%2c0.4%2c12.2%2c0.4%2c18.1%2c0.4C133.4%2c392.9%2c140.1%2c392.6%2c147%2c392.2L147%2c392.2z M138.5%2c110.4c2.8-12.6%2c6.2-25%2c10.3-37.2c-16-4.5-32.5-7.9-48.9-10.3c-1.5%2c16.7-1.7%2c33.4-0.7%2c49.8C112.2%2c111.2%2c125.5%2c110.5%2c138.5%2c110.4L138.5%2c110.4z M163%2c109.5c11.2-7%2c23.1-13%2c35.7-17.5c-9-4-17.9-7.8-27.4-11.3C168.2%2c90%2c165.4%2c99.9%2c163%2c109.5z M44.5%2c316.9l-1.3-0.9c-13.2-8.3-25.9-17.6-37.8-27.3l-3.4-2.8l0.7%2c4.5c3.2%2c24.6%2c10%2c48.3%2c20%2c71l1.7%2c4.1l1.5-4.1c4.9-14.5%2c11.1-28.9%2c17.9-42.5L44.5%2c316.9z M77.1%2c74.7C62%2c89.9%2c48.5%2c106.7%2c37.6%2c125.3c13.2-3.9%2c26.5-7.1%2c39.9-9.4C76.7%2c102.2%2c76.6%2c88.6%2c77.1%2c74.7z M84.8%2c253.7l0.3-4.4c0.7-15.6%2c3.2-30.8%2c7.9-45.6l1.3-4.3l-3.9%2c2.2c-10%2c5.8-19.9%2c12.2-29.3%2c19l-1.5%2c1l1.1%2c1.5c6.7%2c9.4%2c13.7%2c18.6%2c21.2%2c27.4L84.8%2c253.7zM86.6%2c279.9l-0.7-4.2l-2.5%2c3.8c-7.1%2c10-13.7%2c20.3-19.7%2c31l-0.8%2c1.5l1.5%2c0.8c10.7%2c6%2c22%2c11.5%2c33.3%2c16.5l4.1%2c1.7l-1.9-4.1C93.2%2c311.9%2c88.7%2c296.1%2c86.6%2c279.9L86.6%2c279.9z M36.7%2c218.9l1.3-1.1l-0.8-1.3c-7.9-12.8-15.2-26.3-21.6-40l-1.9-3.8l-1.2%2c4.3c-7.5%2c23.1-11.6%2c47.3-12.2%2c71.6L0%2c253.1l3.3-3.2C13.7%2c238.8%2c25%2c228.1%2c36.7%2c218.9L36.7%2c218.9z' style='fill: green%3b'/%3e %3c/g%3e %3c/svg%3e");
  --anaconda-catalog-chevron-right-icon-image: url("data:image/svg+xml,%3csvg width='16' height='16' viewBox='0 0 16 16' xmlns='http://www.w3.org/2000/svg'%3e %3cpath d='M4.93945 4.06066L8.87883 8.00003L4.93945 11.9394L6.00011 13L11.0001 8.00003L6.00011 3L4.93945 4.06066Z'/%3e %3c/svg%3e");
  --anaconda-catalog-list-icon-image: url("data:image/svg+xml,%3csvg width='16' height='16' viewBox='0 0 16 16' xmlns='http://www.w3.org/2000/svg'%3e %3cpath d='M4 5H1V2H4V5ZM4 7H1V10H4V7ZM4 12H1V15H4V12ZM6 4H15V3H6V4ZM6 9H15V8H6V9ZM6 14H15V13H6V14Z'/%3e %3c/svg%3e");
  --anaconda-catalog-grid-icon-image: url("data:image/svg+xml,%3csvg width='15' height='11' viewBox='0 0 15 11' xmlns='http://www.w3.org/2000/svg'%3e %3cpath d='M1.16659 5.16666H3.66659C4.12492 5.16666 4.49992 4.79166 4.49992 4.33332V0.99999C4.49992 0.541656 4.12492 0.166656 3.66659 0.166656H1.16659C0.708252 0.166656 0.333252 0.541656 0.333252 0.99999V4.33332C0.333252 4.79166 0.708252 5.16666 1.16659 5.16666ZM1.16659 11H3.66659C4.12492 11 4.49992 10.625 4.49992 10.1667V6.83332C4.49992 6.37499 4.12492 5.99999 3.66659 5.99999H1.16659C0.708252 5.99999 0.333252 6.37499 0.333252 6.83332V10.1667C0.333252 10.625 0.708252 11 1.16659 11ZM6.16659 11H8.66659C9.12492 11 9.49992 10.625 9.49992 10.1667V6.83332C9.49992 6.37499 9.12492 5.99999 8.66659 5.99999H6.16659C5.70825 5.99999 5.33325 6.37499 5.33325 6.83332V10.1667C5.33325 10.625 5.70825 11 6.16659 11ZM11.1666 11H13.6666C14.1249 11 14.4999 10.625 14.4999 10.1667V6.83332C14.4999 6.37499 14.1249 5.99999 13.6666 5.99999H11.1666C10.7083 5.99999 10.3333 6.37499 10.3333 6.83332V10.1667C10.3333 10.625 10.7083 11 11.1666 11ZM6.16659 5.16666H8.66659C9.12492 5.16666 9.49992 4.79166 9.49992 4.33332V0.99999C9.49992 0.541656 9.12492 0.166656 8.66659 0.166656H6.16659C5.70825 0.166656 5.33325 0.541656 5.33325 0.99999V4.33332C5.33325 4.79166 5.70825 5.16666 6.16659 5.16666ZM10.3333 0.99999V4.33332C10.3333 4.79166 10.7083 5.16666 11.1666 5.16666H13.6666C14.1249 5.16666 14.4999 4.79166 14.4999 4.33332V0.99999C14.4999 0.541656 14.1249 0.166656 13.6666 0.166656H11.1666C10.7083 0.166656 10.3333 0.541656 10.3333 0.99999Z'/%3e %3c/svg%3e");
  --anaconda-catalog-search-icon-image: url("data:image/svg+xml,%3csvg width='18' height='18' viewBox='0 0 18 18' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath d='M12.5 11H11.71L11.43 10.73C12.63 9.33 13.25 7.42 12.91 5.39C12.44 2.61 10.12 0.39 7.32 0.05C3.09 -0.47 -0.47 3.09 0.05 7.32C0.39 10.12 2.61 12.44 5.39 12.91C7.42 13.25 9.33 12.63 10.73 11.43L11 11.71V12.5L15.25 16.75C15.66 17.16 16.33 17.16 16.74 16.75C17.15 16.34 17.15 15.67 16.74 15.26L12.5 11ZM6.5 11C4.01 11 2 8.99 2 6.5C2 4.01 4.01 2 6.5 2C8.99 2 11 4.01 11 6.5C11 8.99 8.99 11 6.5 11Z' fill='%23323232'/%3e %3c/svg%3e");
  --anaconda-catalog-share-icon-image: url("data:image/svg+xml,%3csvg width='24' height='24' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cg clip-path='url(%23clip0_289_5006)'%3e %3cpath d='M18 16.08C17.24 16.08 16.56 16.38 16.04 16.85L8.91 12.7C8.96 12.47 9 12.24 9 12C9 11.76 8.96 11.53 8.91 11.3L15.96 7.19C16.5 7.69 17.21 8 18 8C19.66 8 21 6.66 21 5C21 3.34 19.66 2 18 2C16.34 2 15 3.34 15 5C15 5.24 15.04 5.47 15.09 5.7L8.04 9.81C7.5 9.31 6.79 9 6 9C4.34 9 3 10.34 3 12C3 13.66 4.34 15 6 15C6.79 15 7.5 14.69 8.04 14.19L15.16 18.35C15.11 18.56 15.08 18.78 15.08 19C15.08 20.61 16.39 21.92 18 21.92C19.61 21.92 20.92 20.61 20.92 19C20.92 17.39 19.61 16.08 18 16.08Z' fill='%23323232'/%3e %3c/g%3e %3cdefs%3e %3cclipPath id='clip0_289_5006'%3e %3crect width='24' height='24' fill='white'/%3e %3c/clipPath%3e %3c/defs%3e %3c/svg%3e");
  --notebook-shared-link-copy-icon-image: url("data:image/svg+xml,%3csvg width='19' height='22' viewBox='0 0 19 22' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath d='M13 0H2C0.9 0 0 0.9 0 2V15C0 15.55 0.45 16 1 16C1.55 16 2 15.55 2 15V3C2 2.45 2.45 2 3 2H13C13.55 2 14 1.55 14 1C14 0.45 13.55 0 13 0ZM13.59 4.59L18.42 9.42C18.79 9.79 19 10.3 19 10.83V20C19 21.1 18.1 22 17 22H5.99C4.89 22 4 21.1 4 20L4.01 6C4.01 4.9 4.9 4 6 4H12.17C12.7 4 13.21 4.21 13.59 4.59ZM13 11H17.5L12 5.5V10C12 10.55 12.45 11 13 11Z' fill='%238C94A6'/%3e %3c/svg%3e");
  --notebook-shared-badge-copy-icon-image: url("data:image/svg+xml,%3csvg width='21' height='24' viewBox='0 0 21 24' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath d='M17.64 0H3.36C2.46887 0 1.61424 0.337142 0.984121 0.937258C0.353999 1.53737 0 2.35131 0 3.2L0 20.8C0 21.6487 0.353999 22.4626 0.984121 23.0627C1.61424 23.6629 2.46887 24 3.36 24H17.64C18.5311 24 19.3858 23.6629 20.0159 23.0627C20.646 22.4626 21 21.6487 21 20.8V3.2C21 2.35131 20.646 1.53737 20.0159 0.937258C19.3858 0.337142 18.5311 0 17.64 0V0ZM12.8226 7.976L13.713 7.128L16.611 9.884L13.713 12.644L12.8226 11.796L14.8302 9.884L12.8226 7.976ZM10.836 6.656L12.0498 6.98L10.3152 12.88L9.1014 12.556L10.836 6.656ZM5.754 2.2C5.90019 2.19602 6.04431 2.23367 6.16791 2.30813C6.29151 2.38258 6.38897 2.49047 6.44784 2.61797C6.5067 2.74548 6.52429 2.88681 6.49834 3.02389C6.4724 3.16097 6.40411 3.28756 6.3022 3.38746C6.20029 3.48737 6.06941 3.55605 5.92629 3.58472C5.78317 3.61339 5.63433 3.60075 5.49879 3.54841C5.36326 3.49607 5.24721 3.40641 5.16548 3.2909C5.08375 3.17539 5.04006 3.03929 5.04 2.9C5.03992 2.71776 5.11447 2.54268 5.2478 2.41196C5.38113 2.28125 5.56273 2.20521 5.754 2.2ZM7.287 7.128L8.1774 7.976L6.1698 9.884L8.1774 11.796L7.287 12.644L4.389 9.884L7.287 7.128ZM2.52 2.9C2.52006 2.76071 2.56375 2.62461 2.64548 2.5091C2.72721 2.3936 2.84326 2.30394 2.97879 2.25159C3.11433 2.19925 3.26317 2.1866 3.40629 2.21528C3.54941 2.24395 3.68029 2.31263 3.7822 2.41254C3.88411 2.51244 3.9524 2.63903 3.97834 2.77611C4.00429 2.91319 3.9867 3.05452 3.92784 3.18203C3.86897 3.30953 3.77151 3.41742 3.64791 3.49187C3.52431 3.56633 3.38019 3.60398 3.234 3.6C3.04273 3.59479 2.86113 3.51875 2.7278 3.38804C2.59447 3.25732 2.51992 3.08224 2.52 2.9ZM9.849 20.4H3.129V19.6H9.849V20.4ZM17.829 18H3.129V17.2H17.829V18Z' fill='%238C94A6'/%3e %3c/svg%3e");
  --notebook-shared-link-icon-image: url("data:image/svg+xml,%3csvg width='20' height='20' viewBox='0 0 20 20' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath fill-rule='evenodd' clip-rule='evenodd' d='M5 12.5C3.61719 12.5 2.5 11.3828 2.5 10C2.5 8.61719 3.61719 7.5 5 7.5H9.32812C8.46484 6.00781 6.85156 5 5 5C2.23828 5 0 7.23828 0 10C0 12.7617 2.23828 15 5 15C6.84766 15 8.46484 13.9922 9.32812 12.5H5ZM15 15C13.1484 15 11.5351 13.9922 10.6718 12.5H15C16.3828 12.5 17.5 11.3828 17.5 10C17.5 8.61719 16.3789 7.5 15 7.5H10.6718C11.5351 6.00781 13.1523 5 15 5C17.7617 5 20 7.23828 20 10C20 12.7617 17.7617 15 15 15ZM3.75 10C3.75 9.30859 4.30859 8.75 5 8.75H15C15.6914 8.75 16.25 9.30859 16.25 10C16.25 10.6914 15.6914 11.25 15 11.25H5C4.30859 11.25 3.75 10.6914 3.75 10Z' fill='%236D5BF6'/%3e %3c/svg%3e");
  --notebook-shared-code-icon-image: url("data:image/svg+xml,%3csvg width='20' height='20' viewBox='0 0 20 20' fill='none' xmlns='http://www.w3.org/2000/svg'%3e %3cpath d='M7.52214 4.77167C7.85553 4.45217 7.86595 3.92777 7.54992 3.59438C7.23389 3.26099 6.70603 3.25057 6.37263 3.5666L0.260462 9.40094C0.0937663 9.55722 0 9.77254 0 10.0017C0 10.2309 0.0937663 10.4463 0.256989 10.606L6.36916 16.4404C6.70255 16.7599 7.23042 16.746 7.54645 16.4126C7.86248 16.0792 7.85206 15.5513 7.51867 15.2353L2.04202 10.0017L7.52214 4.77167ZM12.4848 4.77167L17.9615 10.0017L12.4813 15.2318C12.1479 15.5513 12.1375 16.0757 12.4536 16.4091C12.7696 16.7425 13.2974 16.7529 13.6308 16.4369L19.743 10.6025C19.9062 10.4463 20 10.2275 20 9.99827C20 9.76906 19.9062 9.55375 19.743 9.394L13.6308 3.55965C13.2974 3.24015 12.7696 3.25404 12.4536 3.58744C12.1375 3.92083 12.1479 4.4487 12.4813 4.76472L12.4848 4.77167Z' fill='black'/%3e %3c/svg%3e");
  --anaconda-catalog-icon-default-square-width-and-height: 30px;
  --anaconda-catalog-icon-compact-square-width-and-height: 23px;
  --anaconda-catalog-font-family: inter, serif;
  --anaconda-notebook-shared-link-secondary-color: #6d5bf6;
  --anaconda-notebook-shared-link-secondary-color-hover: #9286f7;
}

/* Inter */
@font-face {
  font-family: Inter;
  src: local('Inter'),
    url(http://localhost:8888/lab/extensions/@anaconda/share_notebook/static/4a772478a65263896de7.ttf) format('truetype');
}

.anaconda-data-catalog-icon {
  align-items: center;
  display: flex;
  height: 52px;
  width: 52px;
  background-color: var(--anaconda-catalog-primary-color);
  -webkit-mask-image: var(--anaconda-catalog-icon-image);
  mask-image: var(--anaconda-catalog-icon-image);
}

div[class$='-sectionHeader'] .anaconda-data-catalog-icon {
  width: 32px;
  height: 32px;
}

.anaconda-catalog-launcher {
  font-family: var(--anaconda-catalog-font-family);
  display: flex;
  flex-direction: column;
  padding: 42px;
  color: var(--anaconda-catalog-default-color);
  height: calc(100% - 29px);
  overflow: auto;
}

.anaconda-catalog-launcher-list {
  display: grid;
  grid-template-columns: repeat(auto-fill, 291px);
  justify-content: space-between;
  margin-left: -28px;
}

.anaconda-catalog-launcher-header {
  margin-bottom: 25px;
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: flex-start;
}

.anaconda-catalog-launcher-subheader h3,
.anaconda-catalog-launcher-header h3,
.anaconda-catalog-launcher-header h4 {
  font-weight: 500;
  font-size: 34px;
  line-height: 38px;
  margin: 0;
}

.anaconda-catalog-launcher-header h4 {
  font-weight: 300;
  font-size: 18px;
  line-height: 20px;
  margin-top: 10px;
}

.anaconda-catalog-launcher-subheader {
  margin-bottom: 25px;
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: flex-start;
}

.anaconda-catalog-chevron-right-icon {
  width: 16px;
  height: 16px;
  display: flex;
  align-items: center;
  background-color: var(--anaconda-catalog-primary-color);
  -webkit-mask-image: var(--anaconda-catalog-chevron-right-icon-image);
  mask-image: var(--anaconda-catalog-chevron-right-icon-image);
}

.anaconda-catalog-share-icon {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  background-color: var(--jp-ui-font-color1);
  -webkit-mask-image: var(--anaconda-catalog-share-icon-image);
  mask-image: var(--anaconda-catalog-share-icon-image);
}

.anaconda-catalog-launcher-toolbar {
  width: 100%;
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  margin-bottom: 28px;
  align-items: center;
}

.anaconda-catalog-share {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.anaconda-catalog-share label {
  text-align: center;
  font-weight: 400;
  font-size: 10px;
  line-height: 16px;
  text-transform: uppercase;
}

.anaconda-catalog-share-button,
.display-toggle {
  box-sizing: border-box;
  display: flex;
  width: 36px;
  height: 36px;
  border-radius: 8px;
  align-items: center;
  justify-content: center;
}

.display-toggle {
  margin-left: 5px;
}

.anaconda-catalog-share-button,
.anaconda-catalog-launcher-toolbar .display-toggle.active {
  border: 0.5px solid var(--anaconda-catalog-tile-border);
  background: var(--anaconda-catalog-tile-toggle-background);
}

.anaconda-catalog-share-button:hover,
.anaconda-catalog-launcher-toolbar .display-toggle:hover {
  background: var(--anaconda-catalog-tile-toggle-background-hover);
}

.anaconda-catalog-search-field {
  display: flex;
  flex-grow: 1;
  font-size: 18px;
  line-height: 20px;
  align-items: center;
}

.anaconda-catalog-search-icon {
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  background-color: var(--jp-ui-font-color1);
  -webkit-mask-image: var(--anaconda-catalog-search-icon-image);
  mask-image: var(--anaconda-catalog-search-icon-image);
  margin-left: 14px;
  margin-right: -30px;
}

.anaconda-catalog-search-field input {
  font-weight: 400;
  font-size: 12px;
  line-height: 14px;
  height: 34px;
  box-sizing: border-box;
  width: 50%;
  border-radius: 17px;
  border: 1px solid var(--jp-border-color3);
  padding: 0 10px 0 44px;
}

.anaconda-catalog-tile {
  box-sizing: border-box;
  width: 263px;
  height: 190px;
  margin: 0 0 28px 28px;
  background: var(--anaconda-catalog-tile-background);
  border: 0.5px solid var(--anaconda-catalog-tile-border);
  border-radius: 4px;
  position: relative;
  padding: 17px 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.anaconda-catalog-error,
.anaconda-catalog-no-results {
  background: var(--anaconda-catalog-tile-background);
  border: 0.5px solid var(--anaconda-catalog-tile-border);
  border-radius: 4px;
  position: relative;
  padding: 17px 36px;
  display: flex;
  flex-direction: column;
}

.anaconda-catalog-tile:hover,
.anaconda-catalog-tile:active,
.anaconda-catalog-tile:focus {
  filter: var(--anaconda-catalog-tile-shadow);
}

.anaconda-catalog-tile:hover {
  cursor: pointer;
}

.anaconda-catalog-tile-left-border {
  position: absolute;
  width: 5px;
  height: 190px;
  background: var(--anaconda-catalog-primary-color);
  border-radius: 4px 0 0 4px;
  top: -1px;
  left: -1px;
}

.anaconda-catalog-error .anaconda-catalog-tile-left-border,
.anaconda-catalog-no-results .anaconda-catalog-tile-left-border {
  position: absolute;
  width: 5px;
  height: calc(100% + 1px);
  background: var(--anaconda-catalog-primary-color);
}

.anaconda-catalog-error .anaconda-catalog-tile-left-border {
  background: var(--anaconda-catalog-alert-color);
}

.anaconda-catalog-tile-name {
  font-weight: 400;
  font-size: 22px;
  line-height: 23px;
  margin: 8px;
}

.anaconda-catalog-tile-description {
  box-sizing: border-box;
  margin: 8px 0;
  font-weight: 300;
  font-size: 12px;
  line-height: 15px;
  text-align: start;
  width: 263px;
  padding: 0 26px;
  overflow: hidden;
  text-overflow: ellipsis;
}

.anaconda-catalog-learn-more {
  font-weight: 400;
  font-size: 11px;
  line-height: 15px;
  color: var(--anaconda-catalog-primary-color);
  white-space: nowrap;
}

.anaconda-catalog-learn-more:hover {
  color: var(--anaconda-catalog-primary-color-hover);
}

.anaconda-catalog-learn-more::before {
  content: ' ';
}

.anaconda-catalog-tile-action {
  font-weight: 600;
  font-size: 12px;
  line-height: 15px;
  color: var(--anaconda-catalog-primary-color);
  width: 100%;
  position: absolute;
  bottom: 17px;
}

.anaconda-catalog-tile:hover .anaconda-catalog-tile-action,
.anaconda-catalog-tile:active .anaconda-catalog-tile-action,
.anaconda-catalog-tile:focus .anaconda-catalog-tile-action {
  color: var(--jp-ui-font-color1);
}

.anaconda-catalog-tile-icon {
  width: var(--anaconda-catalog-icon-default-square-width-and-height);
  height: var(--anaconda-catalog-icon-default-square-width-and-height);
  display: flex;
  align-items: center;
  background-color: var(--anaconda-catalog-primary-color);
  -webkit-mask-image: var(--anaconda-catalog-default-icon-image);
  mask-image: var(--anaconda-catalog-default-icon-image);
}

.anaconda-catalog-list-icon {
  width: 16px;
  height: 16px;
  display: flex;
  align-items: center;
  background-color: var(--jp-ui-font-color1);
  -webkit-mask-image: var(--anaconda-catalog-list-icon-image);
  mask-image: var(--anaconda-catalog-list-icon-image);
}

.anaconda-catalog-grid-icon {
  width: 15px;
  height: 11px;
  display: flex;
  align-items: center;
  background-color: var(--jp-ui-font-color1);
  -webkit-mask-image: var(--anaconda-catalog-grid-icon-image);
  mask-image: var(--anaconda-catalog-grid-icon-image);
}

.anaconda-catalog-launcher-list.compact {
  grid-template-columns: 100%;
  margin: 0;
}

.anaconda-catalog-launcher-list.compact .anaconda-catalog-tile {
  height: auto;
  width: 100%;
  margin: 0 20px 10px 0;
  padding: 5px;
  flex-direction: row;
  text-align: start;
}

.anaconda-catalog-launcher-list.compact .anaconda-catalog-tile-left-border {
  height: calc(100% + 1px);
}

.anaconda-catalog-launcher-list.compact .anaconda-catalog-tile-name-and-description {
  display: flex;
  flex: 1 1 auto;
  flex-direction: column;
  overflow: hidden;
}

.anaconda-catalog-launcher-list.compact .anaconda-catalog-tile-name {
  font-weight: 400;
  font-size: 14px;
  line-height: 14px;
  margin: 0 0 2px;
}

.anaconda-catalog-launcher-list.compact .anaconda-catalog-tile-description {
  font-weight: 300;
  font-size: 10px;
  width: auto;
  margin: 0;
  padding: 0;
  white-space: nowrap;
}

.anaconda-catalog-launcher-list.compact .anaconda-catalog-tile-action {
  width: auto;
  position: static;
  bottom: auto;
  white-space: nowrap;
  margin-right: 10px;
}

.anaconda-catalog-launcher-list.compact .anaconda-catalog-tile-icon {
  min-width: var(--anaconda-catalog-icon-compact-square-width-and-height);
  min-height: var(--anaconda-catalog-icon-compact-square-width-and-height);
  width: var(--anaconda-catalog-icon-compact-square-width-and-height);
  height: var(--anaconda-catalog-icon-compact-square-width-and-height);
  -webkit-mask-size: var(--anaconda-catalog-icon-compact-square-width-and-height) var(--anaconda-catalog-icon-compact-square-width-and-height);
  mask-size: var(--anaconda-catalog-icon-compact-square-width-and-height) var(--anaconda-catalog-icon-compact-square-width-and-height);
  margin: -2px 10px 0;
}

.anaconda-catalog-launcher-list.compact .anaconda-catalog-learn-more {
  margin-left: 10px;
}

.catalog-launcher-open-notebook {
  background: var(--anaconda-catalog-open-notebook-background);
  border: 1px solid var(--anaconda-catalog-open-notebook-border);
  border-radius: 8px;
  padding: 8px 14px;
  margin: 0;
  white-space: nowrap;
  font-weight: 700;
  font-size: 16px;
  line-height: 24px;
}

.catalog-launcher-open-notebook:hover,
.catalog-launcher-open-notebook:focus {
  background-color: var(--anaconda-catalog-open-notebook-hover);
  border-color: var(--anaconda-catalog-open-notebook-hover);
  text-decoration: none;
}

.catalog-view-back {
  display: block;
  font-size: 12px;
  line-height: 14px;
  font-weight: 400px;
  margin-bottom: 13px;
  color: var(--anaconda-catalog-primary-color);
}

.catalog-view-back:hover {
  color: var(--anaconda-catalog-primary-color-hover);
}

.catalog-view-title {
  font-size: 34px;
  line-height: 38px;
}

.anaconda-catalog-metadata {
  box-sizing: border-box;
  margin: 0 0 28px;
  background: var(--jp-layout-color1);
  border: 0.5px solid var(--jp-border-color3);
  border-radius: 4px;
  position: relative;
  padding: 17px 36px;
  display: flex;
  flex-direction: column;
  word-wrap: break-word;
}

.grouped-fields {
  display: flex;
  flex-flow: row wrap;
}

.grouped-fields .anaconda-catalog-metadata-field {
  flex-grow: 1;
  flex-basis: 50%;
}

.anaconda-catalog-metadata-field:not(:first-child) {
  margin-top: 10px;
}

.grouped-fields .anaconda-catalog-metadata-field:not(:first-child) {
  margin-top: 0;
}

.anaconda-catalog-metadata label {
  font-weight: 600;
  clear: both;
  display: block;
}

.anaconda-catalog-data-url,
.anaconda-catalog-data-url:hover {
  color: var(--anaconda-catalog-primary-color);
}

.anaconda-catalog-data-url:hover {
  color: var(--anaconda-catalog-primary-color-hover);
  text-decoration: none;
}

.anaconda-catalog-tile-left-border-2 {
  position: absolute;
  width: 5px;
  height: calc(100% + 1px);
  background: var(--anaconda-catalog-primary-color);
  border-radius: 4px 0 0 4px;
  top: -1px;
  left: -1px;
}

.notebook-shared-link {
  display: flex;
  flex-direction: column;
  align-items: left;
  width: 465px;
  border-top: 1px solid var(--jp-border-color2);
}

.notebook-shared-subheader {
	font-size: 16px;
    font-style: normal;
    font-weight: 700;
    margin-bottom: unset;
}

.notebook-shared-link-copy {
  display: flex;
  flex-direction: row;
  align-items: center;
}

.notebook-shared-badge h4 {
  margin: 20px 0 10px;
}

.notebook-shared-link-copy input,
.notebook-shared-badge input {
  width: 100%;
  margin-right: 10px;
  border-radius: 4px;
  height: 36px;
  background: var(--jp-input-background);
}

.notebook-shared-badge input {
	border-radius: 4px;
    border: 1px solid var(--gray-400, #8C94A6);
    background: var(--gray-200, #CED2D9);
    height: 57px;
}

.notebook-shared-link-copy-button {
  box-sizing: content-box;
  display: flex;
  width: 19px;
  height: 22px;
  align-items: right;
  justify-content: right;
}

.jp-Button.jp-ToolbarButtonComponent.share-notebook-button {
  display: flex;
  align-items: center;
  font-family: var(--anaconda-catalog-font-family);
  font-weight: 700;
  font-size: 27px;
  line-height: 24px;
  background-color: var(--anaconda-catalog-open-notebook-background);
  border-radius: 4px;
  padding: 0 10px;
}

.jp-Button.jp-ToolbarButtonComponent.share-notebook-button * {
  color: var(--md-grey-900) !important;
}

button.jp-Button.jp-ToolbarButtonComponent.share-notebook-button:hover {
  background-color: var(--anaconda-catalog-open-notebook-hover);
}

.jp-Button.jp-ToolbarButtonComponent.share-notebook-button span {
  display: flex;
  align-items: center;
  flex-direction: row-reverse;
}

.jp-Button.jp-ToolbarButtonComponent.share-notebook-button span .jp-ToolbarButtonComponent-icon {
  margin-left: 4px;
}

.jp-Button.jp-ToolbarButtonComponent.share-notebook-button span span svg g path {
  fill: var(--md-grey-900);
}

.notebook-shared-code-icon,
.notebook-shared-link-icon {
  width: 20px;
  height: 20px;
  display: flex;
  align-items: center;
  background-color: var(--anaconda-notebook-shared-link-secondary-color);
  margin-right: 8px;
}

.notebook-shared-link-icon {
  -webkit-mask-image: var(--notebook-shared-link-icon-image);
  mask-image: var(--notebook-shared-link-icon-image);
}

.notebook-shared-code-icon {
  -webkit-mask-image: var(--notebook-shared-code-icon-image);
  mask-image: var(--notebook-shared-code-icon-image);
}

button:hover .notebook-shared-code-icon,
button:hover .notebook-shared-link-icon {
  background-color: var(--anaconda-notebook-shared-link-secondary-color-hover);
}

.notebook-shared-link-copy-icon {
  width: 19px;
  height: 22px;
  display: flex;
  align-items: right;
  background-color: var(--jp-layout-color4);
  -webkit-mask-image: var(--notebook-shared-link-copy-icon-image);
  mask-image: var(--notebook-shared-link-copy-icon-image);
  cursor: pointer;
}

.notebook-shared-badge {
  display: flex;
  flex-direction: column;
  align-items: left;
  margin-top: 20px;
  border-top: 1px solid var(--jp-border-color2);
  border-bottom: 1px solid var(--jp-border-color2);
  padding-bottom: 20px;
  margin-bottom: 10px;
}

.notebook-shared-badge-copy {
  display: flex;
  flex-direction: row;
  align-items: left;
}

button.notebook-shared-copy-button {
  background: var(--jp-layout-color1);
  border: 1px solid var(--anaconda-notebook-shared-link-secondary-color);
  color: var(--anaconda-notebook-shared-link-secondary-color);
  border-radius: 4px;
  padding: 8px 14px;
  margin: 13px 0 0;
  white-space: nowrap;
  font-weight: 700;
  font-size: 16px;
  line-height: 24px;
  width: fit-content;
  display: flex;
  align-items: center;
  height: 40px;
}

button.notebook-shared-copy-button:hover {
  background-color: var(--jp-layout-color2);
  border-color: var(--anaconda-notebook-shared-link-secondary-color-hover);
  color: var(--anaconda-notebook-shared-link-secondary-color-hover);
  text-decoration: none;
  cursor: pointer;
}

.notebook-shared-badge-copy-button {
  box-sizing: content-box;
  display: flex;
  width: 21px;
  height: 24px;
  align-items: right;
  justify-content: right;
  align-self: center;
}

.notebook-shared-badge-copy-icon {
  width: 21px;
  height: 24px;
  display: flex;
  align-items: right;
  background-color: var(--jp-layout-color4);
  -webkit-mask-image: var(--notebook-shared-badge-copy-icon-image);
  mask-image: var(--notebook-shared-badge-copy-icon-image);
  cursor: pointer;
}

.notebook-shared-badge-image {
  margin-top: 12px;
  font-size: 11px;
  line-height: 20px;
}

/* When enabled, moves the data catalog to the top of the launcher */
.data-catalog-launcher-position-experiment .jp-Launcher-content {
  display: flex;
  flex-direction: column;
}

.data-catalog-launcher-position-experiment .jp-Launcher-content .jp-Launcher-cwd {
  order: -2;
}

.data-catalog-launcher-position-experiment .jp-Launcher-content .jp-Launcher-section:has([data-category='Anaconda']) {
  order: -1;
}

.notebook-shared-modal .jp-Dialog-header {
	font-size: 24px;
	font-style: normal;
	font-weight: 600;
}

.notebook-shared-modal .jp-Dialog-content {
    min-height: 585px;
    width: 515px;
}

.notebook-shared-modal .jp-Dialog-body {
	overflow: hidden;
}

.notebook-shared-modal .jp-Dialog-footer button {
	color: #06262D;
    font-size: 16px;
    font-style: normal;
    font-weight: 700;
}
</style><style>.Toastify__toast-container{z-index:9999;-webkit-transform:translateZ(9999px);position:fixed;padding:4px;width:320px;box-sizing:border-box;color:#fff}.Toastify__toast-container--top-left{top:1em;left:1em}.Toastify__toast-container--top-center{top:1em;left:50%;transform:translateX(-50%)}.Toastify__toast-container--top-right{top:1em;right:1em}.Toastify__toast-container--bottom-left{bottom:1em;left:1em}.Toastify__toast-container--bottom-center{bottom:1em;left:50%;transform:translateX(-50%)}.Toastify__toast-container--bottom-right{bottom:1em;right:1em}@media only screen and (max-width:480px){.Toastify__toast-container{width:100vw;padding:0;left:0;margin:0}.Toastify__toast-container--top-center,.Toastify__toast-container--top-left,.Toastify__toast-container--top-right{top:0;transform:translateX(0)}.Toastify__toast-container--bottom-center,.Toastify__toast-container--bottom-left,.Toastify__toast-container--bottom-right{bottom:0;transform:translateX(0)}.Toastify__toast-container--rtl{right:0;left:auto}}.Toastify__toast{position:relative;min-height:64px;box-sizing:border-box;margin-bottom:1rem;padding:8px;border-radius:4px;box-shadow:0 1px 10px 0 rgba(0,0,0,.1),0 2px 15px 0 rgba(0,0,0,.05);display:-ms-flexbox;display:flex;-ms-flex-pack:justify;justify-content:space-between;max-height:800px;overflow:hidden;font-family:sans-serif;cursor:pointer;direction:ltr}.Toastify__toast--rtl{direction:rtl}.Toastify__toast--dark{background:#121212;color:#fff}.Toastify__toast--default{background:#fff;color:#aaa}.Toastify__toast--info{background:#3498db}.Toastify__toast--success{background:#07bc0c}.Toastify__toast--warning{background:#f1c40f}.Toastify__toast--error{background:#e74c3c}.Toastify__toast-body{margin:auto 0;-ms-flex:1 1 auto;flex:1 1 auto;padding:6px}.Toastify--animate{animation-fill-mode:both;animation-duration:.7s}@media only screen and (max-width:480px){.Toastify__toast{margin-bottom:0;border-radius:0}}.Toastify__close-button{color:#fff;background:transparent;outline:none;border:none;padding:0;cursor:pointer;opacity:.7;transition:.3s ease;-ms-flex-item-align:start;align-self:flex-start}.Toastify__close-button--default{color:#000;opacity:.3}.Toastify__close-button>svg{fill:currentColor;height:16px;width:14px}.Toastify__close-button:focus,.Toastify__close-button:hover{opacity:1}@keyframes Toastify__trackProgress{0%{transform:scaleX(1)}to{transform:scaleX(0)}}.Toastify__progress-bar{position:absolute;bottom:0;left:0;width:100%;height:5px;z-index:9999;opacity:.7;background-color:hsla(0,0%,100%,.7);transform-origin:left}.Toastify__progress-bar--animated{animation:Toastify__trackProgress linear 1 forwards}.Toastify__progress-bar--controlled{transition:transform .2s}.Toastify__progress-bar--rtl{right:0;left:auto;transform-origin:right}.Toastify__progress-bar--default{background:linear-gradient(90deg,#4cd964,#5ac8fa,#007aff,#34aadc,#5856d6,#ff2d55)}.Toastify__progress-bar--dark{background:#bb86fc}@keyframes Toastify__bounceInRight{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(3000px,0,0)}60%{opacity:1;transform:translate3d(-25px,0,0)}75%{transform:translate3d(10px,0,0)}90%{transform:translate3d(-5px,0,0)}to{transform:none}}@keyframes Toastify__bounceOutRight{20%{opacity:1;transform:translate3d(-20px,0,0)}to{opacity:0;transform:translate3d(2000px,0,0)}}@keyframes Toastify__bounceInLeft{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(-3000px,0,0)}60%{opacity:1;transform:translate3d(25px,0,0)}75%{transform:translate3d(-10px,0,0)}90%{transform:translate3d(5px,0,0)}to{transform:none}}@keyframes Toastify__bounceOutLeft{20%{opacity:1;transform:translate3d(20px,0,0)}to{opacity:0;transform:translate3d(-2000px,0,0)}}@keyframes Toastify__bounceInUp{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(0,3000px,0)}60%{opacity:1;transform:translate3d(0,-20px,0)}75%{transform:translate3d(0,10px,0)}90%{transform:translate3d(0,-5px,0)}to{transform:translateZ(0)}}@keyframes Toastify__bounceOutUp{20%{transform:translate3d(0,-10px,0)}40%,45%{opacity:1;transform:translate3d(0,20px,0)}to{opacity:0;transform:translate3d(0,-2000px,0)}}@keyframes Toastify__bounceInDown{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(0,-3000px,0)}60%{opacity:1;transform:translate3d(0,25px,0)}75%{transform:translate3d(0,-10px,0)}90%{transform:translate3d(0,5px,0)}to{transform:none}}@keyframes Toastify__bounceOutDown{20%{transform:translate3d(0,10px,0)}40%,45%{opacity:1;transform:translate3d(0,-20px,0)}to{opacity:0;transform:translate3d(0,2000px,0)}}.Toastify__bounce-enter--bottom-left,.Toastify__bounce-enter--top-left{animation-name:Toastify__bounceInLeft}.Toastify__bounce-enter--bottom-right,.Toastify__bounce-enter--top-right{animation-name:Toastify__bounceInRight}.Toastify__bounce-enter--top-center{animation-name:Toastify__bounceInDown}.Toastify__bounce-enter--bottom-center{animation-name:Toastify__bounceInUp}.Toastify__bounce-exit--bottom-left,.Toastify__bounce-exit--top-left{animation-name:Toastify__bounceOutLeft}.Toastify__bounce-exit--bottom-right,.Toastify__bounce-exit--top-right{animation-name:Toastify__bounceOutRight}.Toastify__bounce-exit--top-center{animation-name:Toastify__bounceOutUp}.Toastify__bounce-exit--bottom-center{animation-name:Toastify__bounceOutDown}@keyframes Toastify__zoomIn{0%{opacity:0;transform:scale3d(.3,.3,.3)}50%{opacity:1}}@keyframes Toastify__zoomOut{0%{opacity:1}50%{opacity:0;transform:scale3d(.3,.3,.3)}to{opacity:0}}.Toastify__zoom-enter{animation-name:Toastify__zoomIn}.Toastify__zoom-exit{animation-name:Toastify__zoomOut}@keyframes Toastify__flipIn{0%{transform:perspective(400px) rotateX(90deg);animation-timing-function:ease-in;opacity:0}40%{transform:perspective(400px) rotateX(-20deg);animation-timing-function:ease-in}60%{transform:perspective(400px) rotateX(10deg);opacity:1}80%{transform:perspective(400px) rotateX(-5deg)}to{transform:perspective(400px)}}@keyframes Toastify__flipOut{0%{transform:perspective(400px)}30%{transform:perspective(400px) rotateX(-20deg);opacity:1}to{transform:perspective(400px) rotateX(90deg);opacity:0}}.Toastify__flip-enter{animation-name:Toastify__flipIn}.Toastify__flip-exit{animation-name:Toastify__flipOut}@keyframes Toastify__slideInRight{0%{transform:translate3d(110%,0,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideInLeft{0%{transform:translate3d(-110%,0,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideInUp{0%{transform:translate3d(0,110%,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideInDown{0%{transform:translate3d(0,-110%,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideOutRight{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(110%,0,0)}}@keyframes Toastify__slideOutLeft{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(-110%,0,0)}}@keyframes Toastify__slideOutDown{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(0,500px,0)}}@keyframes Toastify__slideOutUp{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(0,-500px,0)}}.Toastify__slide-enter--bottom-left,.Toastify__slide-enter--top-left{animation-name:Toastify__slideInLeft}.Toastify__slide-enter--bottom-right,.Toastify__slide-enter--top-right{animation-name:Toastify__slideInRight}.Toastify__slide-enter--top-center{animation-name:Toastify__slideInDown}.Toastify__slide-enter--bottom-center{animation-name:Toastify__slideInUp}.Toastify__slide-exit--bottom-left,.Toastify__slide-exit--top-left{animation-name:Toastify__slideOutLeft}.Toastify__slide-exit--bottom-right,.Toastify__slide-exit--top-right{animation-name:Toastify__slideOutRight}.Toastify__slide-exit--top-center{animation-name:Toastify__slideOutUp}.Toastify__slide-exit--bottom-center{animation-name:Toastify__slideOutDown}</style><style>.Toastify__toast {
  min-height: unset;
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  color: var(--jp-ui-font-color1);
  border-width: var(--jp-border-width);
  border-radius: var(--jp-border-radius);
  border-color: var(--jp-border-color1);
  background: var(--jp-layout-color1);
  box-shadow: var(--jp-elevation-z4);
  cursor: default;
}

.Toastify__toast-body > svg.svg-inline--fa {
  margin-top: 0.2em;
}

.Toastify__toast--error {
  border-top: 5px solid var(--jp-error-color1);
}

.Toastify__toast--warning {
  border-top: 5px solid var(--jp-warn-color1);
}

.Toastify__toast--info {
  border-top: 5px solid var(--jp-info-color1);
}

.Toastify__toast--success {
  border-top: 5px solid var(--jp-success-color1);
}

.Toastify__toast--default.jp-toast-in-progress {
  border-top: 5px solid var(--jp-layout-color1);
}

.jp-toast-buttonBar {
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  flex: 0 0 auto;
}

.jp-toast-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-toast-button {
  border: solid var(--jp-border-width) var(--jp-border-color2);
  font-size: var(--jp-ui-font-size0);
  margin-top: 1px;
  margin-bottom: 1px;
  margin-right: 0px;
  margin-left: 3px;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color2);
}

.jp-toast-button:hover {
  background-color: var(--jp-layout-color3);
}
</style><style>/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-InterfaceSwitcher {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.jp-InterfaceSwitcher .lm-MenuBar-itemIcon svg {
  vertical-align: sub;
}

.jp-nb-interface-switcher-button > .jp-ToolbarButtonComponent {
  flex-direction: row-reverse;
}

.jp-nb-interface-switcher-button
  > .jp-ToolbarButtonComponent
  > .jp-ToolbarButtonComponent-icon {
  padding-left: 3px;
}
</style><style>/* Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

 .jupyter-widgets-disconnected::before {
    content: "\f127"; /* chain-broken */
    display: inline-block;
    font: normal normal normal 14px/1 FontAwesome;
    font-size: inherit;
    text-rendering: auto;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    color: #d9534f;
    padding: 3px;
    align-self: flex-start;
}
</style><style>/* This file has code derived from PhosphorJS CSS files, as noted below. The license for this PhosphorJS code is:

Copyright (c) 2014-2017, PhosphorJS Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/*
 * The following section is derived from https://github.com/phosphorjs/phosphor/blob/23b9d075ebc5b73ab148b6ebfc20af97f85714c4/packages/widgets/style/tabbar.css 
 * We've scoped the rules so that they are consistent with exactly our code.
 */

.jupyter-widgets.widget-tab > .p-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}


.jupyter-widgets.widget-tab > .p-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
}


.jupyter-widgets.widget-tab > .p-TabBar[data-orientation='vertical'] {
  flex-direction: column;
}


.jupyter-widgets.widget-tab > .p-TabBar > .p-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}


.jupyter-widgets.widget-tab > .p-TabBar[data-orientation='horizontal'] > .p-TabBar-content {
  flex-direction: row;
}


.jupyter-widgets.widget-tab > .p-TabBar[data-orientation='vertical'] > .p-TabBar-content {
  flex-direction: column;
}


.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
}


.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabIcon,
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}


.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}


.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab.p-mod-hidden {
  display: none !important;
}


.jupyter-widgets.widget-tab > .p-TabBar.p-mod-dragging .p-TabBar-tab {
  position: relative;
}


.jupyter-widgets.widget-tab > .p-TabBar.p-mod-dragging[data-orientation='horizontal'] .p-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}


.jupyter-widgets.widget-tab > .p-TabBar.p-mod-dragging[data-orientation='vertical'] .p-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}


.jupyter-widgets.widget-tab > .p-TabBar.p-mod-dragging .p-TabBar-tab.p-mod-dragging {
  transition: none;
}

/* End tabbar.css */
</style><style>/* Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
 * We assume that the CSS variables in
 * https://github.com/jupyterlab/jupyterlab/blob/master/src/default-theme/variables.css
 * have been defined.
 */

:root {
    --jp-widgets-color: var(--jp-content-font-color1);
    --jp-widgets-label-color: var(--jp-widgets-color);
    --jp-widgets-readout-color: var(--jp-widgets-color);
    --jp-widgets-font-size: var(--jp-ui-font-size1);
    --jp-widgets-margin: 2px;
    --jp-widgets-inline-height: 28px;
    --jp-widgets-inline-width: 300px;
    --jp-widgets-inline-width-short: calc(var(--jp-widgets-inline-width) / 2 - var(--jp-widgets-margin));
    --jp-widgets-inline-width-tiny: calc(var(--jp-widgets-inline-width-short) / 2 - var(--jp-widgets-margin));
    --jp-widgets-inline-margin: 4px; /* margin between inline elements */
    --jp-widgets-inline-label-width: 80px;
    --jp-widgets-border-width: var(--jp-border-width);
    --jp-widgets-vertical-height: 200px;
    --jp-widgets-horizontal-tab-height: 24px;
    --jp-widgets-horizontal-tab-width: 144px;
    --jp-widgets-horizontal-tab-top-border: 2px;
    --jp-widgets-progress-thickness: 20px;
    --jp-widgets-container-padding: 15px;
    --jp-widgets-input-padding: 4px;
    --jp-widgets-radio-item-height-adjustment: 8px;
    --jp-widgets-radio-item-height: calc(var(--jp-widgets-inline-height) - var(--jp-widgets-radio-item-height-adjustment));
    --jp-widgets-slider-track-thickness: 4px;
    --jp-widgets-slider-border-width: var(--jp-widgets-border-width);
    --jp-widgets-slider-handle-size: 16px;
    --jp-widgets-slider-handle-border-color: var(--jp-border-color1);
    --jp-widgets-slider-handle-background-color: var(--jp-layout-color1);
    --jp-widgets-slider-active-handle-color: var(--jp-brand-color1);
    --jp-widgets-menu-item-height: 24px;
    --jp-widgets-dropdown-arrow: url("data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjIuMSwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAxOCAxOCIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgMTggMTg7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4KPHN0eWxlIHR5cGU9InRleHQvY3NzIj4KCS5zdDB7ZmlsbDpub25lO30KPC9zdHlsZT4KPHBhdGggZD0iTTUuMiw1LjlMOSw5LjdsMy44LTMuOGwxLjIsMS4ybC00LjksNWwtNC45LTVMNS4yLDUuOXoiLz4KPHBhdGggY2xhc3M9InN0MCIgZD0iTTAtMC42aDE4djE4SDBWLTAuNnoiLz4KPC9zdmc+Cg");
    --jp-widgets-input-color: var(--jp-ui-font-color1);
    --jp-widgets-input-background-color: var(--jp-layout-color1);
    --jp-widgets-input-border-color: var(--jp-border-color1);
    --jp-widgets-input-focus-border-color: var(--jp-brand-color2);
    --jp-widgets-input-border-width: var(--jp-widgets-border-width);
    --jp-widgets-disabled-opacity: 0.6;

    /* From Material Design Lite */
    --md-shadow-key-umbra-opacity: 0.2;
    --md-shadow-key-penumbra-opacity: 0.14;
    --md-shadow-ambient-shadow-opacity: 0.12;
}

.jupyter-widgets {
    margin: var(--jp-widgets-margin);
    box-sizing: border-box;
    color: var(--jp-widgets-color);
    overflow: visible;
}

.jupyter-widgets.jupyter-widgets-disconnected::before {
    line-height: var(--jp-widgets-inline-height);
    height: var(--jp-widgets-inline-height);
}

.jp-Output-result > .jupyter-widgets {
    margin-left: 0;
    margin-right: 0;
}

/* vbox and hbox */

.widget-inline-hbox {
    /* Horizontal widgets */
    box-sizing: border-box;
    display: flex;
    flex-direction: row;
    align-items: baseline;
}

.widget-inline-vbox {
    /* Vertical Widgets */
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.widget-box {
    box-sizing: border-box;
    display: flex;
    margin: 0;
    overflow: auto;
}

.widget-gridbox {
    box-sizing: border-box;
    display: grid;
    margin: 0;
    overflow: auto;
}

.widget-hbox {
    flex-direction: row;
}

.widget-vbox {
    flex-direction: column;
}

/* General Button Styling */

.jupyter-button {
    padding-left: 10px;
    padding-right: 10px;
    padding-top: 0px;
    padding-bottom: 0px;
    display: inline-block;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-align: center;
    font-size: var(--jp-widgets-font-size);
    cursor: pointer;

    height: var(--jp-widgets-inline-height);
    border: 0px solid;
    line-height: var(--jp-widgets-inline-height);
    box-shadow: none;

    color: var(--jp-ui-font-color1);
    background-color: var(--jp-layout-color2);
    border-color: var(--jp-border-color2);
    border: none;
    user-select: none;
}

.jupyter-button i.fa {
    margin-right: var(--jp-widgets-inline-margin);
    pointer-events: none;
}

.jupyter-button:empty:before {
    content: "\200b"; /* zero-width space */
}

.jupyter-widgets.jupyter-button:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

.jupyter-button i.fa.center {
    margin-right: 0;
}

.jupyter-button:hover:enabled, .jupyter-button:focus:enabled {
    /* MD Lite 2dp shadow */
    box-shadow: 0 2px 2px 0 rgba(0, 0, 0, var(--md-shadow-key-penumbra-opacity)),
                0 3px 1px -2px rgba(0, 0, 0, var(--md-shadow-key-umbra-opacity)),
                0 1px 5px 0 rgba(0, 0, 0, var(--md-shadow-ambient-shadow-opacity));
}

.jupyter-button:active, .jupyter-button.mod-active {
    /* MD Lite 4dp shadow */
    box-shadow: 0 4px 5px 0 rgba(0, 0, 0, var(--md-shadow-key-penumbra-opacity)),
                0 1px 10px 0 rgba(0, 0, 0, var(--md-shadow-ambient-shadow-opacity)),
                0 2px 4px -1px rgba(0, 0, 0, var(--md-shadow-key-umbra-opacity));
    color: var(--jp-ui-font-color1);
    background-color: var(--jp-layout-color3);
}

.jupyter-button:focus:enabled {
    outline: 1px solid var(--jp-widgets-input-focus-border-color);
}

/* Button "Primary" Styling */

.jupyter-button.mod-primary {
    color: var(--jp-ui-inverse-font-color1, var(--jp-inverse-ui-font-color1));
    background-color: var(--jp-brand-color1);
}

.jupyter-button.mod-primary.mod-active {
    color: var(--jp-ui-inverse-font-color0, var(--jp-inverse-ui-font-color0));
    background-color: var(--jp-brand-color0);
}

.jupyter-button.mod-primary:active {
    color: var(--jp-ui-inverse-font-color0, var(--jp-inverse-ui-font-color0));
    background-color: var(--jp-brand-color0);
}

/* Button "Success" Styling */

.jupyter-button.mod-success {
    color: var(--jp-ui-inverse-font-color1, var(--jp-inverse-ui-font-color1));
    background-color: var(--jp-success-color1);
}

.jupyter-button.mod-success.mod-active {
    color: var(--jp-ui-inverse-font-color0, var(--jp-inverse-ui-font-color0));
    background-color: var(--jp-success-color0);
}

.jupyter-button.mod-success:active {
    color: var(--jp-ui-inverse-font-color0, var(--jp-inverse-ui-font-color0));
    background-color: var(--jp-success-color0);
}

 /* Button "Info" Styling */

.jupyter-button.mod-info {
    color: var(--jp-ui-inverse-font-color1, var(--jp-inverse-ui-font-color1));
    background-color: var(--jp-info-color1);
}

.jupyter-button.mod-info.mod-active {
    color: var(--jp-ui-inverse-font-color0, var(--jp-inverse-ui-font-color0));
    background-color: var(--jp-info-color0);
}

.jupyter-button.mod-info:active {
    color: var(--jp-ui-inverse-font-color0, var(--jp-inverse-ui-font-color0));
    background-color: var(--jp-info-color0);
}

/* Button "Warning" Styling */

.jupyter-button.mod-warning {
    color: var(--jp-ui-inverse-font-color1, var(--jp-inverse-ui-font-color1));
    background-color: var(--jp-warn-color1);
}

.jupyter-button.mod-warning.mod-active {
    color: var(--jp-ui-inverse-font-color0, var(--jp-inverse-ui-font-color0));
    background-color: var(--jp-warn-color0);
}

.jupyter-button.mod-warning:active {
    color: var(--jp-ui-inverse-font-color0, var(--jp-inverse-ui-font-color0));
    background-color: var(--jp-warn-color0);
}

/* Button "Danger" Styling */

.jupyter-button.mod-danger {
    color: var(--jp-ui-inverse-font-color1, var(--jp-inverse-ui-font-color1));
    background-color: var(--jp-error-color1);
}

.jupyter-button.mod-danger.mod-active {
    color: var(--jp-ui-inverse-font-color0, var(--jp-inverse-ui-font-color0));
    background-color: var(--jp-error-color0);
}

.jupyter-button.mod-danger:active {
    color: var(--jp-ui-inverse-font-color0, var(--jp-inverse-ui-font-color0));
    background-color: var(--jp-error-color0);
}

/* Widget Button, Widget Toggle Button, Widget Upload */

.widget-button, .widget-toggle-button, .widget-upload {
    width: var(--jp-widgets-inline-width-short);
}

/* Widget Label Styling */

/* Override Bootstrap label css */
.jupyter-widgets label {
    margin-bottom: initial;
}

.widget-label-basic {
    /* Basic Label */
    color: var(--jp-widgets-label-color);
    font-size: var(--jp-widgets-font-size);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    line-height: var(--jp-widgets-inline-height);
}

.widget-label {
    /* Label */
    color: var(--jp-widgets-label-color);
    font-size: var(--jp-widgets-font-size);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    line-height: var(--jp-widgets-inline-height);
}

.widget-inline-hbox .widget-label {
    /* Horizontal Widget Label */
    color: var(--jp-widgets-label-color);
    text-align: right;
    margin-right: calc( var(--jp-widgets-inline-margin) * 2 );
    width: var(--jp-widgets-inline-label-width);
    flex-shrink: 0;
}

.widget-inline-vbox .widget-label {
    /* Vertical Widget Label */
    color: var(--jp-widgets-label-color);
    text-align: center;
    line-height: var(--jp-widgets-inline-height);
}

/* Widget Readout Styling */

.widget-readout {
    color: var(--jp-widgets-readout-color);
    font-size: var(--jp-widgets-font-size);
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
    overflow: hidden;
    white-space: nowrap;
    text-align: center;
}

.widget-readout.overflow {
    /* Overflowing Readout */

    /* From Material Design Lite
        shadow-key-umbra-opacity: 0.2;
        shadow-key-penumbra-opacity: 0.14;
        shadow-ambient-shadow-opacity: 0.12;
     */
    -webkit-box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2),
                        0 3px 1px -2px rgba(0, 0, 0, 0.14),
                        0 1px 5px 0 rgba(0, 0, 0, 0.12);

    -moz-box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2),
                     0 3px 1px -2px rgba(0, 0, 0, 0.14),
                     0 1px 5px 0 rgba(0, 0, 0, 0.12);

    box-shadow: 0 2px 2px 0 rgba(0, 0, 0, 0.2),
                0 3px 1px -2px rgba(0, 0, 0, 0.14),
                0 1px 5px 0 rgba(0, 0, 0, 0.12);
}

.widget-inline-hbox .widget-readout {
    /* Horizontal Readout */
    text-align: center;
    max-width: var(--jp-widgets-inline-width-short);
    min-width: var(--jp-widgets-inline-width-tiny);
    margin-left: var(--jp-widgets-inline-margin);
}

.widget-inline-vbox .widget-readout {
    /* Vertical Readout */
    margin-top: var(--jp-widgets-inline-margin);
    /* as wide as the widget */
    width: inherit;
}

/* Widget Checkbox Styling */

.widget-checkbox {
    width: var(--jp-widgets-inline-width);
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
}

.widget-checkbox input[type="checkbox"] {
    margin: 0px calc( var(--jp-widgets-inline-margin) * 2 ) 0px 0px;
    line-height: var(--jp-widgets-inline-height);
    font-size: large;
    flex-grow: 1;
    flex-shrink: 0;
    align-self: center;
}

/* Widget Valid Styling */

.widget-valid {
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
    width: var(--jp-widgets-inline-width-short);
    font-size: var(--jp-widgets-font-size);
}

.widget-valid i:before {
    line-height: var(--jp-widgets-inline-height);
    margin-right: var(--jp-widgets-inline-margin);
    margin-left: var(--jp-widgets-inline-margin);
}

.widget-valid.mod-valid i:before {
    color: green;
}

.widget-valid.mod-invalid i:before {
    color: red;
}

.widget-valid.mod-valid .widget-valid-readout {
    display: none;
}

/* Widget Text and TextArea Stying */

.widget-textarea, .widget-text {
    width: var(--jp-widgets-inline-width);
}

.widget-text input[type="text"], .widget-text input[type="number"], .widget-text input[type="password"] {
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
}

.widget-text input[type="text"]:disabled, .widget-text input[type="number"]:disabled, .widget-text input[type="password"]:disabled, .widget-textarea textarea:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

.widget-text input[type="text"], .widget-text input[type="number"], .widget-text input[type="password"], .widget-textarea textarea {
    box-sizing: border-box;
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    background-color: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    font-size: var(--jp-widgets-font-size);
    flex-grow: 1;
    min-width: 0; /* This makes it possible for the flexbox to shrink this input */
    flex-shrink: 1;
    outline: none !important;
}
    
.widget-text input[type="text"], .widget-text input[type="password"], .widget-textarea textarea {
    padding: var(--jp-widgets-input-padding) calc( var(--jp-widgets-input-padding) *  2);
}

.widget-text input[type="number"] {
    padding: var(--jp-widgets-input-padding) 0 var(--jp-widgets-input-padding) calc(var(--jp-widgets-input-padding) *  2);
}

.widget-textarea textarea {
    height: inherit;
    width: inherit;
}

.widget-text input:focus, .widget-textarea textarea:focus {
    border-color: var(--jp-widgets-input-focus-border-color);
}

/* Widget Slider */

.widget-slider .ui-slider {
    /* Slider Track */
    border: var(--jp-widgets-slider-border-width) solid var(--jp-layout-color3);
    background: var(--jp-layout-color3);
    box-sizing: border-box;
    position: relative;
    border-radius: 0px;
}

.widget-slider .ui-slider .ui-slider-handle {
    /* Slider Handle */
    outline: none !important; /* focused slider handles are colored - see below */
    position: absolute;
    background-color: var(--jp-widgets-slider-handle-background-color);
    border: var(--jp-widgets-slider-border-width) solid var(--jp-widgets-slider-handle-border-color);
    box-sizing: border-box;
    z-index: 1;
    background-image: none; /* Override jquery-ui */
}

/* Override jquery-ui */
.widget-slider .ui-slider .ui-slider-handle:hover, .widget-slider .ui-slider .ui-slider-handle:focus {
    background-color: var(--jp-widgets-slider-active-handle-color);
    border: var(--jp-widgets-slider-border-width) solid var(--jp-widgets-slider-active-handle-color);
}

.widget-slider .ui-slider .ui-slider-handle:active {
    background-color: var(--jp-widgets-slider-active-handle-color);
    border-color: var(--jp-widgets-slider-active-handle-color);
    z-index: 2;
    transform: scale(1.2);
}

.widget-slider  .ui-slider .ui-slider-range {
    /* Interval between the two specified value of a double slider */
    position: absolute;
    background: var(--jp-widgets-slider-active-handle-color);
    z-index: 0;
}

/* Shapes of Slider Handles */

.widget-hslider .ui-slider .ui-slider-handle {
    width: var(--jp-widgets-slider-handle-size);
    height: var(--jp-widgets-slider-handle-size);
    margin-top: calc((var(--jp-widgets-slider-track-thickness) - var(--jp-widgets-slider-handle-size)) / 2 - var(--jp-widgets-slider-border-width));
    margin-left: calc(var(--jp-widgets-slider-handle-size) / -2 + var(--jp-widgets-slider-border-width));
    border-radius: 50%;
    top: 0;
}

.widget-vslider .ui-slider .ui-slider-handle {
    width: var(--jp-widgets-slider-handle-size);
    height: var(--jp-widgets-slider-handle-size);
    margin-bottom: calc(var(--jp-widgets-slider-handle-size) / -2 + var(--jp-widgets-slider-border-width));
    margin-left: calc((var(--jp-widgets-slider-track-thickness) - var(--jp-widgets-slider-handle-size)) / 2 - var(--jp-widgets-slider-border-width));
    border-radius: 50%;
    left: 0;
}

.widget-hslider .ui-slider .ui-slider-range {
    height: calc( var(--jp-widgets-slider-track-thickness) * 2 );
    margin-top: calc((var(--jp-widgets-slider-track-thickness) - var(--jp-widgets-slider-track-thickness) * 2 ) / 2 - var(--jp-widgets-slider-border-width));
}

.widget-vslider .ui-slider .ui-slider-range {
    width: calc( var(--jp-widgets-slider-track-thickness) * 2 );
    margin-left: calc((var(--jp-widgets-slider-track-thickness) - var(--jp-widgets-slider-track-thickness) * 2 ) / 2 - var(--jp-widgets-slider-border-width));
}

/* Horizontal Slider */

.widget-hslider {
    width: var(--jp-widgets-inline-width);
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);

    /* Override the align-items baseline. This way, the description and readout
    still seem to align their baseline properly, and we don't have to have
    align-self: stretch in the .slider-container. */
    align-items: center;
}

.widgets-slider .slider-container {
    overflow: visible;
}

.widget-hslider .slider-container {
    height: var(--jp-widgets-inline-height);
    margin-left: calc(var(--jp-widgets-slider-handle-size) / 2 - 2 * var(--jp-widgets-slider-border-width));
    margin-right: calc(var(--jp-widgets-slider-handle-size) / 2 - 2 * var(--jp-widgets-slider-border-width));
    flex: 1 1 var(--jp-widgets-inline-width-short);
}

.widget-hslider .ui-slider {
    /* Inner, invisible slide div */
    height: var(--jp-widgets-slider-track-thickness);
    margin-top: calc((var(--jp-widgets-inline-height) - var(--jp-widgets-slider-track-thickness)) / 2);
    width: 100%;
}

/* Vertical Slider */

.widget-vbox .widget-label {
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
}

.widget-vslider {
    /* Vertical Slider */
    height: var(--jp-widgets-vertical-height);
    width: var(--jp-widgets-inline-width-tiny);
}

.widget-vslider .slider-container {
    flex: 1 1 var(--jp-widgets-inline-width-short);
    margin-left: auto;
    margin-right: auto;
    margin-bottom: calc(var(--jp-widgets-slider-handle-size) / 2 - 2 * var(--jp-widgets-slider-border-width));
    margin-top: calc(var(--jp-widgets-slider-handle-size) / 2 - 2 * var(--jp-widgets-slider-border-width));
    display: flex;
    flex-direction: column;
}

.widget-vslider .ui-slider-vertical {
    /* Inner, invisible slide div */
    width: var(--jp-widgets-slider-track-thickness);
    flex-grow: 1;
    margin-left: auto;
    margin-right: auto;
}

/* Widget Progress Styling */

.progress-bar {
    -webkit-transition: none;
    -moz-transition: none;
    -ms-transition: none;
    -o-transition: none;
    transition: none;
}

.progress-bar {
    height: var(--jp-widgets-inline-height);
}

.progress-bar {
    background-color: var(--jp-brand-color1);
}

.progress-bar-success {
    background-color: var(--jp-success-color1);
}

.progress-bar-info {
    background-color: var(--jp-info-color1);
}

.progress-bar-warning {
    background-color: var(--jp-warn-color1);
}

.progress-bar-danger {
    background-color: var(--jp-error-color1);
}

.progress {
    background-color: var(--jp-layout-color2);
    border: none;
    box-shadow: none;
}

/* Horisontal Progress */

.widget-hprogress {
    /* Progress Bar */
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
    width: var(--jp-widgets-inline-width);
    align-items: center;

}

.widget-hprogress .progress {
    flex-grow: 1;
    margin-top: var(--jp-widgets-input-padding);
    margin-bottom: var(--jp-widgets-input-padding);
    align-self: stretch;
    /* Override bootstrap style */
    height: initial;
}

/* Vertical Progress */

.widget-vprogress {
    height: var(--jp-widgets-vertical-height);
    width: var(--jp-widgets-inline-width-tiny);
}

.widget-vprogress .progress {
    flex-grow: 1;
    width: var(--jp-widgets-progress-thickness);
    margin-left: auto;
    margin-right: auto;
    margin-bottom: 0;
}

/* Select Widget Styling */

.widget-dropdown {
    height: var(--jp-widgets-inline-height);
    width: var(--jp-widgets-inline-width);
    line-height: var(--jp-widgets-inline-height);
}

.widget-dropdown > select {
    padding-right: 20px;
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    border-radius: 0;
    height: inherit;
    flex: 1 1 var(--jp-widgets-inline-width-short);
    min-width: 0; /* This makes it possible for the flexbox to shrink this input */
    box-sizing: border-box;
    outline: none !important;
    box-shadow: none;
    background-color: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    font-size: var(--jp-widgets-font-size);
    vertical-align: top;
    padding-left: calc( var(--jp-widgets-input-padding) * 2);
	appearance: none;
	-webkit-appearance: none;
	-moz-appearance: none;
    background-repeat: no-repeat;
	background-size: 20px;
	background-position: right center;
    background-image: var(--jp-widgets-dropdown-arrow);
}
.widget-dropdown > select:focus {
    border-color: var(--jp-widgets-input-focus-border-color);
}

.widget-dropdown > select:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

/* To disable the dotted border in Firefox around select controls.
   See http://stackoverflow.com/a/18853002 */
.widget-dropdown > select:-moz-focusring {
    color: transparent;
    text-shadow: 0 0 0 #000;
}

/* Select and SelectMultiple */

.widget-select {
    width: var(--jp-widgets-inline-width);
    line-height: var(--jp-widgets-inline-height);

    /* Because Firefox defines the baseline of a select as the bottom of the
    control, we align the entire control to the top and add padding to the
    select to get an approximate first line baseline alignment. */
    align-items: flex-start;
}

.widget-select > select {
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    background-color: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    font-size: var(--jp-widgets-font-size);
    flex: 1 1 var(--jp-widgets-inline-width-short);
    outline: none !important;
    overflow: auto;
    height: inherit;

    /* Because Firefox defines the baseline of a select as the bottom of the
    control, we align the entire control to the top and add padding to the
    select to get an approximate first line baseline alignment. */
    padding-top: 5px;
}

.widget-select > select:focus {
    border-color: var(--jp-widgets-input-focus-border-color);
}

.wiget-select > select > option {
    padding-left: var(--jp-widgets-input-padding);
    line-height: var(--jp-widgets-inline-height);
    /* line-height doesn't work on some browsers for select options */
    padding-top: calc(var(--jp-widgets-inline-height)-var(--jp-widgets-font-size)/2);
    padding-bottom: calc(var(--jp-widgets-inline-height)-var(--jp-widgets-font-size)/2);
}



/* Toggle Buttons Styling */

.widget-toggle-buttons {
    line-height: var(--jp-widgets-inline-height);
}

.widget-toggle-buttons .widget-toggle-button {
    margin-left: var(--jp-widgets-margin);
    margin-right: var(--jp-widgets-margin);
}

.widget-toggle-buttons .jupyter-button:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

/* Radio Buttons Styling */

.widget-radio {
    width: var(--jp-widgets-inline-width);
    line-height: var(--jp-widgets-inline-height);
}

.widget-radio-box {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    box-sizing: border-box;
    flex-grow: 1;
    margin-bottom: var(--jp-widgets-radio-item-height-adjustment);
}

.widget-radio-box label {
    height: var(--jp-widgets-radio-item-height);
    line-height: var(--jp-widgets-radio-item-height);
    font-size: var(--jp-widgets-font-size);
}

.widget-radio-box input {
    height: var(--jp-widgets-radio-item-height);
    line-height: var(--jp-widgets-radio-item-height);
    margin: 0 calc( var(--jp-widgets-input-padding) * 2 ) 0 1px;
    float: left;
}

/* Color Picker Styling */

.widget-colorpicker {
    width: var(--jp-widgets-inline-width);
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
}

.widget-colorpicker > .widget-colorpicker-input {
    flex-grow: 1;
    flex-shrink: 1;
    min-width: var(--jp-widgets-inline-width-tiny);
}

.widget-colorpicker input[type="color"] {
    width: var(--jp-widgets-inline-height);
    height: var(--jp-widgets-inline-height);
    padding: 0 2px; /* make the color square actually square on Chrome on OS X */
    background: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    border-left: none;
    flex-grow: 0;
    flex-shrink: 0;
    box-sizing: border-box;
    align-self: stretch;
    outline: none !important;
}

.widget-colorpicker.concise input[type="color"] {
    border-left: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
}

.widget-colorpicker input[type="color"]:focus, .widget-colorpicker input[type="text"]:focus {
    border-color: var(--jp-widgets-input-focus-border-color);
}

.widget-colorpicker input[type="text"] {
    flex-grow: 1;
    outline: none !important;
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
    background: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    font-size: var(--jp-widgets-font-size);
    padding: var(--jp-widgets-input-padding) calc( var(--jp-widgets-input-padding) *  2 );
    min-width: 0; /* This makes it possible for the flexbox to shrink this input */
    flex-shrink: 1;
    box-sizing: border-box;
}

.widget-colorpicker input[type="text"]:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

/* Date Picker Styling */

.widget-datepicker {
    width: var(--jp-widgets-inline-width);
    height: var(--jp-widgets-inline-height);
    line-height: var(--jp-widgets-inline-height);
}

.widget-datepicker input[type="date"] {
    flex-grow: 1;
    flex-shrink: 1;
    min-width: 0; /* This makes it possible for the flexbox to shrink this input */
    outline: none !important;
    height: var(--jp-widgets-inline-height);
    border: var(--jp-widgets-input-border-width) solid var(--jp-widgets-input-border-color);
    background-color: var(--jp-widgets-input-background-color);
    color: var(--jp-widgets-input-color);
    font-size: var(--jp-widgets-font-size);
    padding: var(--jp-widgets-input-padding) calc( var(--jp-widgets-input-padding) *  2 );
    box-sizing: border-box;
}

.widget-datepicker input[type="date"]:focus {
    border-color: var(--jp-widgets-input-focus-border-color);
}

.widget-datepicker input[type="date"]:invalid {
    border-color: var(--jp-warn-color1);
}

.widget-datepicker input[type="date"]:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

/* Play Widget */

.widget-play {
    width: var(--jp-widgets-inline-width-short);
    display: flex;
    align-items: stretch;
}

.widget-play .jupyter-button {
    flex-grow: 1;
    height: auto;
}

.widget-play .jupyter-button:disabled {
    opacity: var(--jp-widgets-disabled-opacity);
}

/* Tab Widget */

.jupyter-widgets.widget-tab {
    display: flex;
    flex-direction: column;
}

.jupyter-widgets.widget-tab > .p-TabBar {
    /* Necessary so that a tab can be shifted down to overlay the border of the box below. */
    overflow-x: visible;
    overflow-y: visible;
}

.jupyter-widgets.widget-tab > .p-TabBar > .p-TabBar-content {
    /* Make sure that the tab grows from bottom up */
    align-items: flex-end;
    min-width: 0;
    min-height: 0;
}

.jupyter-widgets.widget-tab > .widget-tab-contents {
    width: 100%;
    box-sizing: border-box;
    margin: 0;
    background: var(--jp-layout-color1);
    color: var(--jp-ui-font-color1);
    border: var(--jp-border-width) solid var(--jp-border-color1);
    padding: var(--jp-widgets-container-padding);
    flex-grow: 1;
    overflow: auto;
}

.jupyter-widgets.widget-tab > .p-TabBar {
    font: var(--jp-widgets-font-size) Helvetica, Arial, sans-serif;
    min-height: calc(var(--jp-widgets-horizontal-tab-height) + var(--jp-border-width));
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {
    flex: 0 1 var(--jp-widgets-horizontal-tab-width);
    min-width: 35px;
    min-height: calc(var(--jp-widgets-horizontal-tab-height) + var(--jp-border-width));
    line-height: var(--jp-widgets-horizontal-tab-height);
    margin-left: calc(-1 * var(--jp-border-width));
    padding: 0px 10px;
    background: var(--jp-layout-color2);
    color: var(--jp-ui-font-color2);
    border: var(--jp-border-width) solid var(--jp-border-color1);
    border-bottom: none;
    position: relative;
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current {
    color: var(--jp-ui-font-color0);
    /* We want the background to match the tab content background */
    background: var(--jp-layout-color1);
    min-height: calc(var(--jp-widgets-horizontal-tab-height) + 2 * var(--jp-border-width));
    transform: translateY(var(--jp-border-width));
    overflow: visible;
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab.p-mod-current:before {
    position: absolute;
    top: calc(-1 * var(--jp-border-width));
    left: calc(-1 * var(--jp-border-width));
    content: '';
    height: var(--jp-widgets-horizontal-tab-top-border);
    width: calc(100% + 2 * var(--jp-border-width));
    background: var(--jp-brand-color1);
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab:first-child {
    margin-left: 0;
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab:hover:not(.p-mod-current) {
    background: var(--jp-layout-color1);
    color: var(--jp-ui-font-color1);
}

.jupyter-widgets.widget-tab > .p-TabBar .p-mod-closable > .p-TabBar-tabCloseIcon {
    margin-left: 4px;
}

/* This font-awesome strategy may not work across FA4 and FA5, but we don't
actually support closable tabs, so it really doesn't matter */
.jupyter-widgets.widget-tab > .p-TabBar .p-mod-closable > .p-TabBar-tabCloseIcon:before {
    font-family: FontAwesome;
    content: '\f00d'; /* close */
}

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabIcon,
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabLabel,
.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tabCloseIcon {
    line-height: var(--jp-widgets-horizontal-tab-height);
}

/* Accordion Widget */

.p-Collapse {
    display: flex;
    flex-direction: column;
    align-items: stretch;
}

.p-Collapse-header {
    padding: var(--jp-widgets-input-padding);
    cursor: pointer;
    color: var(--jp-ui-font-color2);
    background-color: var(--jp-layout-color2);
    border: var(--jp-widgets-border-width) solid var(--jp-border-color1);
    padding: calc(var(--jp-widgets-container-padding) * 2 / 3) var(--jp-widgets-container-padding);
    font-weight: bold;
}

.p-Collapse-header:hover {
    background-color: var(--jp-layout-color1);
    color: var(--jp-ui-font-color1);
}

.p-Collapse-open > .p-Collapse-header {
    background-color: var(--jp-layout-color1);
    color: var(--jp-ui-font-color0);
    cursor: default;
    border-bottom: none;
}

.p-Collapse-contents {
    padding: var(--jp-widgets-container-padding);
    background-color: var(--jp-layout-color1);
    color: var(--jp-ui-font-color1);
    border-left: var(--jp-widgets-border-width) solid var(--jp-border-color1);
    border-right: var(--jp-widgets-border-width) solid var(--jp-border-color1);
    border-bottom: var(--jp-widgets-border-width) solid var(--jp-border-color1);
    overflow: auto;
}

.p-Accordion {
    display: flex;
    flex-direction: column;
    align-items: stretch;
}

.p-Accordion .p-Collapse {
    margin-bottom: 0;
}

.p-Accordion .p-Collapse + .p-Collapse {
    margin-top: 4px;
}



/* HTML widget */

.widget-html, .widget-htmlmath {
    font-size: var(--jp-widgets-font-size);
}

.widget-html > .widget-html-content, .widget-htmlmath > .widget-html-content {
    /* Fill out the area in the HTML widget */
    align-self: stretch;
    flex-grow: 1;
    flex-shrink: 1;
    /* Makes sure the baseline is still aligned with other elements */
    line-height: var(--jp-widgets-inline-height);
    /* Make it possible to have absolutely-positioned elements in the html */
    position: relative;
}


/* Image widget  */

.widget-image {
    max-width: 100%;
    height: auto;
}
</style><style></style><style></style><style>.Toastify__toast-container{z-index:9999;-webkit-transform:translateZ(9999px);position:fixed;padding:4px;width:320px;box-sizing:border-box;color:#fff}.Toastify__toast-container--top-left{top:1em;left:1em}.Toastify__toast-container--top-center{top:1em;left:50%;transform:translateX(-50%)}.Toastify__toast-container--top-right{top:1em;right:1em}.Toastify__toast-container--bottom-left{bottom:1em;left:1em}.Toastify__toast-container--bottom-center{bottom:1em;left:50%;transform:translateX(-50%)}.Toastify__toast-container--bottom-right{bottom:1em;right:1em}@media only screen and (max-width:480px){.Toastify__toast-container{width:100vw;padding:0;left:0;margin:0}.Toastify__toast-container--top-center,.Toastify__toast-container--top-left,.Toastify__toast-container--top-right{top:0;transform:translateX(0)}.Toastify__toast-container--bottom-center,.Toastify__toast-container--bottom-left,.Toastify__toast-container--bottom-right{bottom:0;transform:translateX(0)}.Toastify__toast-container--rtl{right:0;left:auto}}.Toastify__toast{position:relative;min-height:64px;box-sizing:border-box;margin-bottom:1rem;padding:8px;border-radius:4px;box-shadow:0 1px 10px 0 rgba(0,0,0,.1),0 2px 15px 0 rgba(0,0,0,.05);display:-ms-flexbox;display:flex;-ms-flex-pack:justify;justify-content:space-between;max-height:800px;overflow:hidden;font-family:sans-serif;cursor:pointer;direction:ltr}.Toastify__toast--rtl{direction:rtl}.Toastify__toast--dark{background:#121212;color:#fff}.Toastify__toast--default{background:#fff;color:#aaa}.Toastify__toast--info{background:#3498db}.Toastify__toast--success{background:#07bc0c}.Toastify__toast--warning{background:#f1c40f}.Toastify__toast--error{background:#e74c3c}.Toastify__toast-body{margin:auto 0;-ms-flex:1 1 auto;flex:1 1 auto;padding:6px}.Toastify--animate{animation-fill-mode:both;animation-duration:.7s}@media only screen and (max-width:480px){.Toastify__toast{margin-bottom:0;border-radius:0}}.Toastify__close-button{color:#fff;background:transparent;outline:none;border:none;padding:0;cursor:pointer;opacity:.7;transition:.3s ease;-ms-flex-item-align:start;align-self:flex-start}.Toastify__close-button--default{color:#000;opacity:.3}.Toastify__close-button>svg{fill:currentColor;height:16px;width:14px}.Toastify__close-button:focus,.Toastify__close-button:hover{opacity:1}@keyframes Toastify__trackProgress{0%{transform:scaleX(1)}to{transform:scaleX(0)}}.Toastify__progress-bar{position:absolute;bottom:0;left:0;width:100%;height:5px;z-index:9999;opacity:.7;background-color:hsla(0,0%,100%,.7);transform-origin:left}.Toastify__progress-bar--animated{animation:Toastify__trackProgress linear 1 forwards}.Toastify__progress-bar--controlled{transition:transform .2s}.Toastify__progress-bar--rtl{right:0;left:auto;transform-origin:right}.Toastify__progress-bar--default{background:linear-gradient(90deg,#4cd964,#5ac8fa,#007aff,#34aadc,#5856d6,#ff2d55)}.Toastify__progress-bar--dark{background:#bb86fc}@keyframes Toastify__bounceInRight{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(3000px,0,0)}60%{opacity:1;transform:translate3d(-25px,0,0)}75%{transform:translate3d(10px,0,0)}90%{transform:translate3d(-5px,0,0)}to{transform:none}}@keyframes Toastify__bounceOutRight{20%{opacity:1;transform:translate3d(-20px,0,0)}to{opacity:0;transform:translate3d(2000px,0,0)}}@keyframes Toastify__bounceInLeft{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(-3000px,0,0)}60%{opacity:1;transform:translate3d(25px,0,0)}75%{transform:translate3d(-10px,0,0)}90%{transform:translate3d(5px,0,0)}to{transform:none}}@keyframes Toastify__bounceOutLeft{20%{opacity:1;transform:translate3d(20px,0,0)}to{opacity:0;transform:translate3d(-2000px,0,0)}}@keyframes Toastify__bounceInUp{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(0,3000px,0)}60%{opacity:1;transform:translate3d(0,-20px,0)}75%{transform:translate3d(0,10px,0)}90%{transform:translate3d(0,-5px,0)}to{transform:translateZ(0)}}@keyframes Toastify__bounceOutUp{20%{transform:translate3d(0,-10px,0)}40%,45%{opacity:1;transform:translate3d(0,20px,0)}to{opacity:0;transform:translate3d(0,-2000px,0)}}@keyframes Toastify__bounceInDown{0%,60%,75%,90%,to{animation-timing-function:cubic-bezier(.215,.61,.355,1)}0%{opacity:0;transform:translate3d(0,-3000px,0)}60%{opacity:1;transform:translate3d(0,25px,0)}75%{transform:translate3d(0,-10px,0)}90%{transform:translate3d(0,5px,0)}to{transform:none}}@keyframes Toastify__bounceOutDown{20%{transform:translate3d(0,10px,0)}40%,45%{opacity:1;transform:translate3d(0,-20px,0)}to{opacity:0;transform:translate3d(0,2000px,0)}}.Toastify__bounce-enter--bottom-left,.Toastify__bounce-enter--top-left{animation-name:Toastify__bounceInLeft}.Toastify__bounce-enter--bottom-right,.Toastify__bounce-enter--top-right{animation-name:Toastify__bounceInRight}.Toastify__bounce-enter--top-center{animation-name:Toastify__bounceInDown}.Toastify__bounce-enter--bottom-center{animation-name:Toastify__bounceInUp}.Toastify__bounce-exit--bottom-left,.Toastify__bounce-exit--top-left{animation-name:Toastify__bounceOutLeft}.Toastify__bounce-exit--bottom-right,.Toastify__bounce-exit--top-right{animation-name:Toastify__bounceOutRight}.Toastify__bounce-exit--top-center{animation-name:Toastify__bounceOutUp}.Toastify__bounce-exit--bottom-center{animation-name:Toastify__bounceOutDown}@keyframes Toastify__zoomIn{0%{opacity:0;transform:scale3d(.3,.3,.3)}50%{opacity:1}}@keyframes Toastify__zoomOut{0%{opacity:1}50%{opacity:0;transform:scale3d(.3,.3,.3)}to{opacity:0}}.Toastify__zoom-enter{animation-name:Toastify__zoomIn}.Toastify__zoom-exit{animation-name:Toastify__zoomOut}@keyframes Toastify__flipIn{0%{transform:perspective(400px) rotateX(90deg);animation-timing-function:ease-in;opacity:0}40%{transform:perspective(400px) rotateX(-20deg);animation-timing-function:ease-in}60%{transform:perspective(400px) rotateX(10deg);opacity:1}80%{transform:perspective(400px) rotateX(-5deg)}to{transform:perspective(400px)}}@keyframes Toastify__flipOut{0%{transform:perspective(400px)}30%{transform:perspective(400px) rotateX(-20deg);opacity:1}to{transform:perspective(400px) rotateX(90deg);opacity:0}}.Toastify__flip-enter{animation-name:Toastify__flipIn}.Toastify__flip-exit{animation-name:Toastify__flipOut}@keyframes Toastify__slideInRight{0%{transform:translate3d(110%,0,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideInLeft{0%{transform:translate3d(-110%,0,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideInUp{0%{transform:translate3d(0,110%,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideInDown{0%{transform:translate3d(0,-110%,0);visibility:visible}to{transform:translateZ(0)}}@keyframes Toastify__slideOutRight{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(110%,0,0)}}@keyframes Toastify__slideOutLeft{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(-110%,0,0)}}@keyframes Toastify__slideOutDown{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(0,500px,0)}}@keyframes Toastify__slideOutUp{0%{transform:translateZ(0)}to{visibility:hidden;transform:translate3d(0,-500px,0)}}.Toastify__slide-enter--bottom-left,.Toastify__slide-enter--top-left{animation-name:Toastify__slideInLeft}.Toastify__slide-enter--bottom-right,.Toastify__slide-enter--top-right{animation-name:Toastify__slideInRight}.Toastify__slide-enter--top-center{animation-name:Toastify__slideInDown}.Toastify__slide-enter--bottom-center{animation-name:Toastify__slideInUp}.Toastify__slide-exit--bottom-left,.Toastify__slide-exit--top-left{animation-name:Toastify__slideOutLeft}.Toastify__slide-exit--bottom-right,.Toastify__slide-exit--top-right{animation-name:Toastify__slideOutRight}.Toastify__slide-exit--top-center{animation-name:Toastify__slideOutUp}.Toastify__slide-exit--bottom-center{animation-name:Toastify__slideOutDown}</style><style>.Toastify__toast {
  min-height: unset;
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  color: var(--jp-ui-font-color1);
  border-width: var(--jp-border-width);
  border-radius: var(--jp-border-radius);
  border-color: var(--jp-border-color1);
  background: var(--jp-layout-color1);
  box-shadow: var(--jp-elevation-z4);
  cursor: default;
}

.Toastify__toast-body > svg.svg-inline--fa {
  margin-top: 0.2em;
}

.Toastify__toast--error {
  border-top: 5px solid var(--jp-error-color1);
}

.Toastify__toast--warning {
  border-top: 5px solid var(--jp-warn-color1);
}

.Toastify__toast--info {
  border-top: 5px solid var(--jp-info-color1);
}

.Toastify__toast--success {
  border-top: 5px solid var(--jp-success-color1);
}

.Toastify__toast--default.jp-toast-in-progress {
  border-top: 5px solid var(--jp-layout-color1);
}

.jp-toast-buttonBar {
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  flex: 0 0 auto;
}

.jp-toast-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-toast-button {
  border: solid var(--jp-border-width) var(--jp-border-color2);
  font-size: var(--jp-ui-font-size0);
  margin-top: 1px;
  margin-bottom: 1px;
  margin-right: 0px;
  margin-left: 3px;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color2);
}

.jp-toast-button:hover {
  background-color: var(--jp-layout-color3);
}
</style><style>.tippy-box[data-animation=fade][data-state=hidden]{opacity:0}[data-tippy-root]{max-width:calc(100vw - 10px)}.tippy-box{position:relative;background-color:#333;color:#fff;border-radius:4px;font-size:14px;line-height:1.4;white-space:normal;outline:0;transition-property:transform,visibility,opacity}.tippy-box[data-placement^=top]>.tippy-arrow{bottom:0}.tippy-box[data-placement^=top]>.tippy-arrow:before{bottom:-7px;left:0;border-width:8px 8px 0;border-top-color:initial;transform-origin:center top}.tippy-box[data-placement^=bottom]>.tippy-arrow{top:0}.tippy-box[data-placement^=bottom]>.tippy-arrow:before{top:-7px;left:0;border-width:0 8px 8px;border-bottom-color:initial;transform-origin:center bottom}.tippy-box[data-placement^=left]>.tippy-arrow{right:0}.tippy-box[data-placement^=left]>.tippy-arrow:before{border-width:8px 0 8px 8px;border-left-color:initial;right:-7px;transform-origin:center left}.tippy-box[data-placement^=right]>.tippy-arrow{left:0}.tippy-box[data-placement^=right]>.tippy-arrow:before{left:-7px;border-width:8px 8px 8px 0;border-right-color:initial;transform-origin:center right}.tippy-box[data-inertia][data-state=visible]{transition-timing-function:cubic-bezier(.54,1.5,.38,1.11)}.tippy-arrow{width:16px;height:16px;color:#333}.tippy-arrow:before{content:"";position:absolute;border-color:transparent;border-style:solid}.tippy-content{position:relative;padding:5px 9px;z-index:1}</style><style>.jp-VarInspector {
  flex-direction: column;
  overflow: auto;
  font-size: var(--jp-ui-font-size1);
}

.jp-VarInspector-table {
  font-family: monospace;
  border-collapse: collapse;
  margin: auto;
  width: 100%;
  color: var(--jp-content-font-color1);
}

.jp-VarInspector-table td,
.jp-VarInspector-table thead {
  border: 1px solid;
  border-color: var(--jp-layout-color2);
  padding: 8px;
}

.jp-VarInspector-table tr:nth-child(even) {
  background-color: var(--jp-layout-color1);
}

.jp-VarInspector-content tr:hover {
  background-color: var(--jp-layout-color2);
}

.jp-VarInspector-table thead {
  font-size: var(--jp-ui-font-size0);
  text-align: center;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color1);
  font-family: sans-serif;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
}

.jp-VarInspector-title {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-content-font-color1);
  text-align: left;
  padding-left: 10px;
}

.jp-VarInspector-deleteButton {
  text-align: center;
  width: 1em;
}

.jp-VarInspector-inspectButton {
  text-align: center;
  width: 1em;
}

.jp-VarInspector-varName {
  font-weight: 600;
}
</style><style>.jp-PanelPreview-renderOnSave {
  display: flex;
  align-items: center;
}

.jp-PanelPreview-loading {
  background: rgba(0, 0, 0, 0.05);
  pointer-events: none;
  position: relative;
}

.jp-PanelPreview-loading::before {
  content: '';
  display: block;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 80px;
  height: 80px;
  border: 8px solid rgb(48, 112, 146);
  border-left-color: #f3f3f3;
  border-radius: 50%;
  animation: spin 1s infinite linear;
  z-index: 9999;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}
</style><style>.tippy-box[data-animation=fade][data-state=hidden]{opacity:0}[data-tippy-root]{max-width:calc(100vw - 10px)}.tippy-box{position:relative;background-color:#333;color:#fff;border-radius:4px;font-size:14px;line-height:1.4;white-space:normal;outline:0;transition-property:transform,visibility,opacity}.tippy-box[data-placement^=top]>.tippy-arrow{bottom:0}.tippy-box[data-placement^=top]>.tippy-arrow:before{bottom:-7px;left:0;border-width:8px 8px 0;border-top-color:initial;transform-origin:center top}.tippy-box[data-placement^=bottom]>.tippy-arrow{top:0}.tippy-box[data-placement^=bottom]>.tippy-arrow:before{top:-7px;left:0;border-width:0 8px 8px;border-bottom-color:initial;transform-origin:center bottom}.tippy-box[data-placement^=left]>.tippy-arrow{right:0}.tippy-box[data-placement^=left]>.tippy-arrow:before{border-width:8px 0 8px 8px;border-left-color:initial;right:-7px;transform-origin:center left}.tippy-box[data-placement^=right]>.tippy-arrow{left:0}.tippy-box[data-placement^=right]>.tippy-arrow:before{left:-7px;border-width:8px 8px 8px 0;border-right-color:initial;transform-origin:center right}.tippy-box[data-inertia][data-state=visible]{transition-timing-function:cubic-bezier(.54,1.5,.38,1.11)}.tippy-arrow{width:16px;height:16px;color:#333}.tippy-arrow:before{content:"";position:absolute;border-color:transparent;border-style:solid}.tippy-content{position:relative;padding:5px 9px;z-index:1}</style><style>.f12ffasl svg{height:16px;width:auto}.fj2oukb{align-items:center;display:flex}.fj2oukb svg{cursor:pointer;display:block;height:28px;margin:auto;padding:2px 2px 2px 8px;width:auto}.f1vya9e0{align-items:center;display:inline-block;vertical-align:middle}.f1vya9e0 svg{display:block;height:16px;margin:0 auto;width:16px}.f9v3xdu:first-child svg{bottom:1px;margin-left:0px;position:relative}.f9v3xdu:hover{background-color:var(--jp-layout-color2)}.jp-mod-dropTarget.f9v3xdu{background-color:var(--jp-brand-color2);opacity:0.7}.f9v3xdu svg{border-radius:var(--jp-border-radius);cursor:pointer;height:16px;margin:0px 2px;padding:0px 2px;vertical-align:middle;width:16px}.fllbajr{align-items:center;display:inline;height:16px;width:16px}.fllbajr svg{display:block;float:right;height:auto;margin:-2px 0 0 0;width:20px}.f1m36mmi{align-items:center;display:flex}.f1m36mmi svg{display:block;height:auto;margin:0 auto;width:32px}.fvaq30{align-items:center;display:flex}.fvaq30 svg{display:block;height:16px;margin:0 auto;width:16px}.f1qewoua{align-items:center;display:flex;height:14px;margin:0 14px 0 auto}.f1qewoua svg{display:block;height:14px;margin:0 auto;width:14px}.f1ozlkqi{pointer-events:none}.f1ozlkqi svg{height:auto;position:absolute;right:7px;top:5px;width:16px}.f1st5hdn{pointer-events:none}.f1st5hdn svg{height:auto;position:absolute;right:8px;top:5px;width:18px}.f1qis64i svg{align-self:normal;height:24px}</style><style id="jp-NotebookExtension-sideBySideMargins">.jp-mod-sideBySide.jp-Notebook .jp-Notebook-cell {
      margin-left: 10px !important;
      margin-right: 10px !important;}</style><style id="react-tooltip-core-styles" type="text/css">:root{--rt-color-white:#fff;--rt-color-dark:#222;--rt-color-success:#8dc572;--rt-color-error:#be6464;--rt-color-warning:#f0ad4e;--rt-color-info:#337ab7;--rt-opacity:0.9;--rt-transition-show-delay:0.15s;--rt-transition-closing-delay:0.15s}.core-styles-module_tooltip__3vRRp{position:absolute;top:0;left:0;pointer-events:none;opacity:0;will-change:opacity}.core-styles-module_fixed__pcSol{position:fixed}.core-styles-module_arrow__cvMwQ{position:absolute;background:inherit}.core-styles-module_noArrow__xock6{display:none}.core-styles-module_clickable__ZuTTB{pointer-events:auto}.core-styles-module_show__Nt9eE{opacity:var(--rt-opacity);transition:opacity var(--rt-transition-show-delay)ease-out}.core-styles-module_closing__sGnxF{opacity:0;transition:opacity var(--rt-transition-closing-delay)ease-in}</style><style id="react-tooltip-base-styles" type="text/css">
.styles-module_tooltip__mnnfp{padding:8px 16px;border-radius:3px;font-size:90%;width:max-content}.styles-module_arrow__K0L3T{width:8px;height:8px}[class*='react-tooltip__place-top']>.styles-module_arrow__K0L3T{transform:rotate(45deg)}[class*='react-tooltip__place-right']>.styles-module_arrow__K0L3T{transform:rotate(135deg)}[class*='react-tooltip__place-bottom']>.styles-module_arrow__K0L3T{transform:rotate(225deg)}[class*='react-tooltip__place-left']>.styles-module_arrow__K0L3T{transform:rotate(315deg)}.styles-module_dark__xNqje{background:var(--rt-color-dark);color:var(--rt-color-white)}.styles-module_light__Z6W-X{background-color:var(--rt-color-white);color:var(--rt-color-dark)}.styles-module_success__A2AKt{background-color:var(--rt-color-success);color:var(--rt-color-white)}.styles-module_warning__SCK0X{background-color:var(--rt-color-warning);color:var(--rt-color-white)}.styles-module_error__JvumD{background-color:var(--rt-color-error);color:var(--rt-color-white)}.styles-module_info__BWdHW{background-color:var(--rt-color-info);color:var(--rt-color-white)}</style></head>
<body data-notebook="notebooks" data-format="desktop" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light" data-jp-theme-scrollbars="false">

  
  

  
  

  
  

  <script id="jupyter-config-data" type="application/json">
    {"allow_hidden_files": false, "appName": "Jupyter Notebook", "appNamespace": "notebook", "appSettingsDir": "C:/Users/CALEB/anaconda3/share/jupyter/lab/settings", "appUrl": "/lab", "appVersion": "7.0.8", "baseUrl": "/", "buildAvailable": true, "buildCheck": true, "cacheFiles": true, "copyAbsolutePath": false, "devMode": false, "disabledExtensions": [], "exposeAppInBrowser": false, "extensionManager": {"can_install": true, "install_path": "C:\\Users\\CALEB\\anaconda3", "name": "PyPI"}, "extraLabextensionsPath": [], "federated_extensions": [{"extension": "./extension", "load": "static/remoteEntry.36690a2bfa9ee0dd7bc8.js", "mimeExtension": "./mimeExtension", "name": "jupyterlab-plotly"}, {"extension": "./extension", "load": "static/remoteEntry.83218267865631694ab2.js", "name": "@anaconda/assistant", "style": "./style"}, {"extension": "./extension", "load": "static/remoteEntry.235a4eb5f08bc399672c.js", "name": "@anaconda/core", "style": "./style"}, {"extension": "./extension", "load": "static/remoteEntry.3318e43690692165b5ed.js", "name": "@anaconda/panels", "style": "./style"}, {"extension": "./extension", "load": "static/remoteEntry.ec403c961f3e6d85e7bc.js", "name": "@anaconda/share_notebook", "style": "./style"}, {"extension": "./extension", "load": "static/remoteEntry.0dca0b46350f0c25c608.js", "name": "@jupyter-notebook/lab-extension", "style": "./style"}, {"extension": "./extension", "load": "static/remoteEntry.ca1efc27dc965162ca86.js", "name": "@jupyter-widgets/jupyterlab-manager"}, {"extension": "./extension", "load": "static/remoteEntry.cfbccb113f4c7c1d65fe.js", "name": "@lckr/jupyterlab_variableinspector", "style": "./style"}, {"extension": "./extension", "load": "static/remoteEntry.c8393541423362724be3.js", "name": "@pyviz/jupyterlab_pyviz", "style": "./style"}], "frontendUrl": "/", "fullAppUrl": "/lab", "fullLabextensionsUrl": "/lab/extensions", "fullLicensesUrl": "/lab/api/licenses", "fullListingsUrl": "/lab/api/listings", "fullMathjaxUrl": "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js", "fullSettingsUrl": "/lab/api/settings", "fullStaticUrl": "/static/notebook", "fullThemesUrl": "/lab/api/themes", "fullTranslationsApiUrl": "/lab/api/translations", "fullTreeUrl": "/lab/tree", "fullWorkspacesApiUrl": "/lab/api/workspaces", "jupyterConfigDir": "C:\\Users\\CALEB\\.jupyter", "labextensionsPath": ["C:\\Users\\CALEB\\AppData\\Roaming\\jupyter\\labextensions", "C:\\Users\\CALEB\\AppData\\Roaming\\Python\\share\\jupyter\\labextensions", "C:\\Users\\CALEB\\anaconda3\\share\\jupyter\\labextensions", "C:\\ProgramData\\jupyter\\labextensions"], "labextensionsUrl": "/lab/extensions", "licensesUrl": "/lab/api/licenses", "listingsUrl": "/lab/api/listings", "mathjaxConfig": "TeX-AMS_HTML-full,Safe", "nbclassic_enabled": false, "news": {"disabled": false}, "notebookPage": "notebooks", "notebookStartsKernel": true, "notebookVersion": "[2, 14, 1]", "preferredPath": "/", "quitButton": true, "rootUri": "file:///c:/Users/CALEB", "schemasDir": "C:/Users/CALEB/anaconda3/share/jupyter/lab/schemas", "settingsUrl": "/lab/api/settings", "staticDir": "C:/Users/CALEB/anaconda3/Lib/site-packages/notebook/static", "templatesDir": "C:/Users/CALEB/anaconda3/Lib/site-packages/notebook/templates", "terminalsAvailable": true, "themesDir": "C:/Users/CALEB/anaconda3/share/jupyter/lab/themes", "themesUrl": "/lab/api/themes", "token": "133a9f0b76be3aee3f72f6b61f734c183e06a1dce4c6b946", "translationsApiUrl": "/lab/api/translations", "treeUrl": "/lab/tree", "userSettingsDir": "C:/Users/CALEB/.jupyter/lab/user-settings", "virtualDocumentsUri": "file:///c:/Users/CALEB/.virtual_documents", "workspacesApiUrl": "/lab/api/workspaces", "workspacesDir": "C:/Users/CALEB/.jupyter/lab/workspaces", "wsUrl": ""}
  </script>
  <script src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/bundle.js.download" main="index"></script>

  <script type="text/javascript">
    /* Remove token from URL. */
    (function () {
      var parsedUrl = new URL(window.location.href);
      if (parsedUrl.searchParams.get('token')) {
        parsedUrl.searchParams.delete('token');
        window.history.replaceState({ }, '', parsedUrl.href);
      }
    })();
  </script>


<div class="lm-Widget as-notebook-container" id="main" data-direction="top-to-bottom" data-alignment="start" style="min-width: 0px; min-height: 70px;"><div class="lm-Widget lm-Panel lm-SplitPanel" data-orientation="horizontal" data-alignment="start" id="main-split-panel" style="position: absolute; contain: strict; min-width: 0px; min-height: 70px; top: 0px; left: 0px; width: 1280px; height: 638px;"><div class="lm-Widget lm-Panel lm-mod-hidden lm-SplitPanel-child" id="jp-left-stack" role="complementary" style="position: absolute; contain: strict;"><button class="jp-Button jp-SidePanel-collapse lm-Widget" title="Collapse left side panel"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" data-icon="ui-components:close" data-icon-id="0f44268c-bc5d-4ba4-8d4f-e9f87f400810"><g class="jp-icon-none jp-icon-selectable-inverse jp-icon3-hover" fill="none"><circle cx="12" cy="12" r="11"></circle></g><g class="jp-icon3 jp-icon-selectable jp-icon-accent2-hover" fill="#616161"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"></path></g><g class="jp-icon-none jp-icon-busy" fill="none"><circle cx="12" cy="12" r="7"></circle></g></svg></button><div class="lm-Widget lm-Panel lm-StackedPanel" style="min-width: 0px; min-height: 0px;"><div class="lm-Widget jp-SidePanel jp-TableOfContents lm-mod-hidden lm-StackedPanel-child" id="table-of-contents" role="region" aria-label="Table of Contents section" style="position: absolute; contain: strict;"><div class="lm-Widget lm-Panel jp-SidePanel-header"><h2 class="jp-text-truncated lm-Widget">Glaso.ipynb</h2></div><div class="lm-Widget jp-Toolbar jp-SidePanel-toolbar" role="navigation" aria-label="side panel actions" style="min-height: var(--jp-private-toolbar-height);"><div class="lm-Widget jp-CommandToolbarButton jp-toc-numberingButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="toc:display-numbering" title="Show heading number in the document"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 28 28" data-icon="ui-components:numbering"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M4 19H6V19.5H5V20.5H6V21H4V22H7V18H4V19ZM5 10H6V6H4V7H5V10ZM4 13H5.8L4 15.1V16H7V15H5.2L7 12.9V12H4V13ZM9 7V9H23V7H9ZM9 21H23V19H9V21ZM9 15H23V13H9V15Z"></path></g></svg></span></button></div><div class="lm-Widget jp-Toolbar-spacer jp-Toolbar-item"></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="toc:toggle-collapse" title="Expand All Headings"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:expand-all"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M8 2c1 0 11 0 12 0s2 1 2 2c0 1 0 11 0 12s0 2-2 2C20 14 20 4 20 4S10 4 6 4c0-2 1-2 2-2z"></path><path d="M18 8c0-1-1-2-2-2S5 6 4 6s-2 1-2 2c0 1 0 11 0 12s1 2 2 2c1 0 11 0 12 0s2-1 2-2c0-1 0-11 0-12zm-2 0v12H4V8z"></path><path d="M11 10H9v3H6v2h3v3h2v-3h3v-2h-3z"></path></g></svg></span></button></div><div class="lm-Widget jp-ToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-pressed="false" aria-disabled="false" title="More actions…"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" data-icon="ui-components:ellipses"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><circle cx="5" cy="12" r="2"></circle><circle cx="12" cy="12" r="2"></circle><circle cx="19" cy="12" r="2"></circle></g></svg></span></button></div></div><div class="lm-Widget lm-Panel jp-SidePanel-content" role="region" aria-label="side panel content"><div class="lm-Widget jp-TableOfContents-tree"><div class="jp-TableOfContents-placeholder"><div class="jp-TableOfContents-placeholderContent"><h3>No Headings</h3><p>The table of contents shows headings in notebooks and supported files.</p></div></div></div></div></div></div></div><div class="lm-SplitPanel-handle lm-mod-hidden" style="position: absolute; contain: style;"></div><div class="lm-Widget lm-Panel lm-SplitPanel-child" data-direction="top-to-bottom" data-alignment="start" style="position: absolute; contain: strict; min-width: 0px; min-height: 70px; top: 0px; left: 0px; width: 1280px; height: 638px;"><div class="lm-Widget lm-Panel" id="top-panel-wrapper" style="position: absolute; contain: strict; top: 0px; left: 0px; width: 1280px; height: 42px;"><div class="lm-Widget lm-Panel" id="top-panel" role="banner"><div class="lm-Widget jp-skiplink" id="jp-skiplink"><a href="http://localhost:8888/notebooks/Downloads%2FMachine%20learning%20and%20ant%20colony%20optimization%2FACO-Algorithm-For-Solution-Gas-Oil-Ratio-PVT-Correlation%2FIpynbFolder%2FGlaso.ipynb#first-cell" tabindex="1" class="skip-link">Skip to Main</a></div><a href="http://localhost:8888/tree" target="_blank" rel="noopener noreferrer" class="lm-Widget fj2oukb" id="jp-NotebookLogo"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="189" height="51" version="2.0" viewBox="0 0 189 51" data-icon="notebook-ui-components:jupyter" data-icon-id="22a28916-7175-4830-a886-de80472a9014"><title>Jupyter</title><g class="jp-icon3" transform="translate(-1638 -2093)" fill="#616161"><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1688.9 2106.2)" style="mix-blend-mode:normal" xlink:href="#path0_fill"></use></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1705.4 2106.2)" style="mix-blend-mode:normal" xlink:href="#path1_fill"></use></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1730.2 2105.7)" style="mix-blend-mode:normal" xlink:href="#path2_fill"></use></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1752.9 2106.2)" style="mix-blend-mode:normal" xlink:href="#path3_fill"></use></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1775.8 2100)" style="mix-blend-mode:normal" xlink:href="#path4_fill"></use></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1791.8 2105.7)" style="mix-blend-mode:normal" xlink:href="#path5_fill"></use></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1815.8 2105.7)" style="mix-blend-mode:normal" xlink:href="#path6_fill"></use></g></g></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1669.3 2093.3)" fill="#767677" style="mix-blend-mode:normal" xlink:href="#path7_fill"></use></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1639.7 2124)" fill="#F37726" style="mix-blend-mode:normal" xlink:href="#path8_fill"></use></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1639.7 2097.5)" fill="#F37726" style="mix-blend-mode:normal" xlink:href="#path9_fill"></use></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1639.8 2135.8)" fill="#989798" style="mix-blend-mode:normal" xlink:href="#path10_fill"></use></g></g><g style="mix-blend-mode:normal"><g style="mix-blend-mode:normal"><use transform="translate(1638.4 2098.1)" fill="#6F7070" style="mix-blend-mode:normal" xlink:href="#path11_fill"></use></g></g></g></g></g><defs><path id="path0_fill" d="m5.6259 17.928c0 5.1461-0.3925 6.8263-1.4392 8.0666-1.1426 1.0559-2.637 1.6399-4.1867 1.6362l0.39251 3.0612c2.402 0.0333 4.7305-0.8352 6.5331-2.4366 1.8753-1.9529 2.5295-4.6535 2.5295-8.7967v-19.458h-3.8378v17.954l0.00872-0.0264z"></path><path id="path1_fill" d="m17.741 15.623c0 2.2168 0 4.1696 0.1744 5.8498h-3.3581l-0.218-3.5187h-0.0873c-0.7069 1.2298-1.7261 2.2473-2.9526 2.9477-1.2265 0.7005-2.6159 1.0585-4.0252 1.0372-3.3145 0-7.2744-1.8649-7.2744-9.3948v-12.544h3.8378v11.902c0 4.0817 1.2211 6.8262 4.7014 6.8262 1.0917-0.0201 2.1533-0.3647 3.0516-0.9907 0.8984-0.6259 1.5936-1.5053 1.9986-2.5279 0.2328-0.6395 0.351-1.3157 0.3489-1.9969v-13.195h3.8379v15.605h-0.0349z"></path><path id="path2_fill" d="m0.17445 7.5363c0-2.7446-0.087223-4.9613-0.17445-7.0374h3.4453l0.17445 3.677h0.08722c0.75374-1.3174 1.85-2.4022 3.1705-3.137 1.3205-0.73485 2.8149-1.0919 4.322-1.0326 5.0939 0 8.9317 4.3983 8.9317 10.908 0 7.7147-4.6141 11.524-9.5946 11.524-1.2785 0.0541-2.5489-0.2281-3.6867-0.8187-1.1377-0.5907-2.1036-1.4696-2.8028-2.5504h-0.08723v11.656h-3.7942v-23.223l0.008722 0.03519zm3.7942 5.7179c0.01001 0.5349 0.0684 1.0679 0.17444 1.5922 0.31262 1.3003 1.0491 2.4571 2.0914 3.2849 1.0423 0.8279 2.33 1.2788 3.6567 1.2805 4.0559 0 6.4022-3.3691 6.4022-8.2864 0-4.3016-2.2242-7.9786-6.2714-7.9786-1.3611 0.03841-2.6704 0.53457-3.7198 1.4096-1.0494 0.87505-1.7786 2.0788-2.0718 3.4198-0.15373 0.51742-0.24166 1.0524-0.26167 1.5922v3.6858z"></path><path id="path3_fill" d="m4.1606 0 4.6141 12.676c0.47973 1.4163 1.0031 3.1053 1.352 4.3984h0.0872c0.3925-1.2843 0.8286-2.9293 1.352-4.4775l4.1867-12.588h4.0559l-5.7481 15.297c-2.7475 7.3541-4.6141 11.128-7.2308 13.433-1.3222 1.2403-2.9429 2.1106-4.7014 2.5246l-0.95946-3.2811c1.2303-0.4134 2.3704-1.0615 3.3581-1.9089 1.3957-1.1762 2.5006-2.664 3.2273-4.3456 0.15514-0.2904 0.25848-0.606 0.30528-0.9324-0.03445-0.3518-0.12272-0.696-0.26167-1.0205l-7.7978-19.766h4.1867l-0.02616-0.0087967z"></path><path id="path4_fill" d="m7.0215 0v6.1577h5.4864v2.9733h-5.4864v11.559c0 2.639 0.7414 4.1696 2.8784 4.1696 0.74972 0.0118 1.4976-0.077 2.2242-0.2639l0.1744 2.9293c-1.0915 0.3877-2.2451 0.5667-3.4017 0.5278-0.76701 0.0474-1.535-0.0744-2.2506-0.357-0.71554-0.2826-1.3614-0.7191-1.8925-1.2792-1.0903-1.1523-1.4828-3.0612-1.4828-5.5858v-11.7h-3.2709v-2.9733h3.2709v-5.1461l3.7506-1.0116z"></path><path id="path5_fill" d="m3.6285 11.928c0.08723 5.278 3.4017 7.4508 7.2308 7.4508 2.0019 0.0634 3.9934-0.3149 5.8353-1.1084l0.6542 2.7886c-2.2069 0.9347-4.585 1.3874-6.9779 1.3283-6.4894 0-10.371-4.3456-10.371-10.82 0-6.4743 3.7506-11.568 9.8912-11.568 6.8819 0 8.7223 6.1577 8.7223 10.107-0.0065 0.6091-0.0501 1.2172-0.1308 1.8209h-14.854zm11.243-2.7885c0.0436-2.4807-1.003-6.3424-5.3119-6.3424-3.8814 0-5.5736 3.633-5.8789 6.3424h11.199-0.0087z"></path><path id="path6_fill" d="m0.17445 7.1785c0-2.5246-0.043612-4.6974-0.17445-6.6943h3.3581l0.13083 4.2136h0.17445c0.95946-2.8853 3.2709-4.6974 5.8353-4.6974 0.36765-0.0054214 0.73435 0.038958 1.0902 0.13195v3.677c-0.4296-0.09447-0.86861-0.13874-1.3083-0.13195-2.7039 0-4.6141 2.0848-5.1375 5.0053-0.10796 0.60107-0.16631 1.2101-0.17445 1.8209v11.436h-3.7942v-14.761z"></path><path id="path7_fill" d="m5.8935 2.844c0.02536 0.58765-0.12268 1.1697-0.42538 1.6724-0.3027 0.50277-0.74647 0.9037-1.2752 1.1521-0.52869 0.24837-1.1186 0.333-1.6949 0.2432-0.57639-0.08981-1.1134-0.35001-1.5432-0.7477-0.42973-0.39768-0.73284-0.91498-0.87099-1.4864-0.13815-0.57146-0.10513-1.1714 0.094877-1.7239 0.20001-0.55254 0.55803-1.0328 1.0288-1.3801 0.47072-0.34728 1.033-0.54595 1.6157-0.57087 0.78063-0.033384 1.5425 0.24712 2.1182 0.77988 0.57569 0.53276 0.91814 1.2742 0.95211 2.0614z"></path><path id="path8_fill" d="m18.265 7.1341c-7.8501 0-14.706-2.8765-18.265-7.1341 1.3254 3.8204 3.7956 7.1308 7.0686 9.473 3.2731 2.3422 7.1871 3.6004 11.2 3.6004s7.9273-1.2582 11.2-3.6004c3.273-2.3422 5.7432-5.6526 7.0686-9.473-3.5675 4.2576-10.423 7.1341-18.273 7.1341z"></path><path id="path9_fill" d="m18.273 5.9393c7.8502 0 14.706 2.8765 18.265 7.1341-1.3254-3.8204-3.7956-7.1308-7.0686-9.473-3.2731-2.3422-7.1871-3.6004-11.2-3.6004s-7.9273 1.2582-11.2 3.6004c-3.273 2.3422-5.7432 5.6526-7.0686 9.473 3.5674-4.2488 10.423-7.1341 18.273-7.1341z"></path><path id="path10_fill" d="m7.4279 3.5834c0.03219 0.74092-0.15434 1.4748-0.53596 2.1088s-0.94118 1.1394-1.6078 1.4525c-0.66664 0.31303-1.4104 0.41954-2.1371 0.30603s-1.4037-0.44193-1.9452-0.94368c-0.54151-0.50175-0.92323-1.1543-1.0968-1.8749-0.17359-0.72067-0.13125-1.4771 0.12166-2.1735 0.25291-0.69639 0.70502-1.3014 1.2991-1.7385 0.59407-0.4371 1.3034-0.68659 2.0381-0.7169 0.98278-0.040537 1.9414 0.31357 2.6657 0.9847 0.72432 0.67112 1.1552 1.6045 1.1983 2.5955z"></path><path id="path11_fill" d="m2.2747 4.3963c-0.43108 0.01879-0.858-0.09184-1.2267-0.31786-0.36872-0.22603-0.66266-0.55729-0.84462-0.95187s-0.24375-0.83473-0.17756-1.2648c0.066192-0.43001 0.25739-0.83055 0.5494-1.1509 0.29201-0.32037 0.6717-0.54618 1.091-0.64882 0.41931-0.10265 0.85939-0.077531 1.2645 0.072176 0.40515 0.14971 0.75716 0.41727 1.0115 0.76882 0.2543 0.35156 0.39947 0.7713 0.41713 1.2061 0.02364 0.58191-0.18257 1.1495-0.57338 1.5783-0.39081 0.42878-0.93431 0.6837-1.5113 0.70883z"></path></defs></svg></a><div class="lm-Widget jp-Toolbar" id="jp-top-bar" style="min-height: var(--jp-private-toolbar-height);"><div class="lm-Widget jp-Toolbar-item" id="jp-title" style="margin-left: 10px;"><h1>Glaso</h1></div><div class="lm-Widget jp-NotebookCheckpoint jp-Toolbar-item" id="id-67fe5471-e09b-47a7-87f9-04e96f154a45">Last Checkpoint: 8 minutes ago</div><div class="lm-Widget jp-Toolbar-spacer jp-Toolbar-item"></div><div class="lm-Widget jp-NotebookKernelLogo jp-Toolbar-item"><img src="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/logo-64x64.png" title="Python 3 (ipykernel)"></div></div></div></div><div class="lm-Widget lm-Panel" id="menu-panel-wrapper" style="position: absolute; contain: strict; top: 42px; left: 0px; width: 1280px; height: 28px;"><div class="lm-Widget lm-Panel" id="menu-panel" role="navigation"><div class="lm-Widget lm-MenuBar jp-scrollbar-tiny" id="jp-MainMenu"><ul class="lm-MenuBar-content" role="menubar"><li tabindex="-1" role="menuitem" aria-haspopup="true" class="lm-MenuBar-item"><div class="lm-MenuBar-itemIcon"></div><div class="lm-MenuBar-itemLabel">File</div></li><li tabindex="-1" role="menuitem" aria-haspopup="true" class="lm-MenuBar-item"><div class="lm-MenuBar-itemIcon"></div><div class="lm-MenuBar-itemLabel">Edit</div></li><li tabindex="-1" role="menuitem" aria-haspopup="true" class="lm-MenuBar-item"><div class="lm-MenuBar-itemIcon"></div><div class="lm-MenuBar-itemLabel">View</div></li><li tabindex="-1" role="menuitem" aria-haspopup="true" class="lm-MenuBar-item"><div class="lm-MenuBar-itemIcon"></div><div class="lm-MenuBar-itemLabel">Run</div></li><li tabindex="-1" role="menuitem" aria-haspopup="true" class="lm-MenuBar-item"><div class="lm-MenuBar-itemIcon"></div><div class="lm-MenuBar-itemLabel">Kernel</div></li><li tabindex="-1" role="menuitem" aria-haspopup="true" class="lm-MenuBar-item"><div class="lm-MenuBar-itemIcon"></div><div class="lm-MenuBar-itemLabel">Settings</div></li><li tabindex="0" role="menuitem" aria-haspopup="true" class="lm-MenuBar-item"><div class="lm-MenuBar-itemIcon"></div><div class="lm-MenuBar-itemLabel">Help</div></li></ul></div><div class="lm-Widget jp-NotebookSpacer" id="id-bfb470ac-8728-43a1-8beb-372cde57d897"></div><div class="lm-Widget jp-NotebookKernelStatus jp-NotebookKernelStatus-fade"></div><div class="lm-Widget"><button class="jp-NotebookTrustedStatus" title="JavaScript enabled for notebook display" style="cursor: help;">Trusted</button></div></div></div><div class="lm-Widget" id="spacer-widget-top" style="position: absolute; contain: strict; top: 70px; left: 0px; width: 1280px; height: 0px;"></div><div class="lm-Widget lm-Panel" id="main-panel" role="main" style="position: absolute; contain: strict; top: 70px; left: 0px; width: 1280px; height: 568px;"><div class="lm-Widget jp-MainAreaWidget jp-NotebookPanel jp-Document jp-mod-searchable" id="id-04007424-d7a3-4d2f-8330-e7a59d316127" data-direction="top-to-bottom" data-alignment="start" style="min-width: 300px; min-height: 29px;"><div class="lm-Widget jp-Toolbar jp-NotebookPanel-toolbar" role="navigation" aria-label="notebook actions" style="position: absolute; contain: strict; top: 0px; left: 0px; width: 1280px; height: 29px; min-height: var(--jp-private-toolbar-height);"><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="docmanager:save" title="Save and create checkpoint (Ctrl+S)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" data-icon="ui-components:save"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M17 3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V7l-4-4zm-5 16c-1.66 0-3-1.34-3-3s1.34-3 3-3 3 1.34 3 3-1.34 3-3 3zm3-10H5V5h10v4z"></path></g></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:insert-cell-below" title="Insert a cell below (B)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:add"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"></path></g></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:cut-cell" title="Cut this cell (X)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" data-icon="ui-components:cut"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M9.64 7.64c.23-.5.36-1.05.36-1.64 0-2.21-1.79-4-4-4S2 3.79 2 6s1.79 4 4 4c.59 0 1.14-.13 1.64-.36L10 12l-2.36 2.36C7.14 14.13 6.59 14 6 14c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4c0-.59-.13-1.14-.36-1.64L12 14l7 7h3v-1L9.64 7.64zM6 8c-1.1 0-2-.89-2-2s.9-2 2-2 2 .89 2 2-.9 2-2 2zm0 12c-1.1 0-2-.89-2-2s.9-2 2-2 2 .89 2 2-.9 2-2 2zm6-7.5c-.28 0-.5-.22-.5-.5s.22-.5.5-.5.5.22.5.5-.22.5-.5.5zM19 3l-6 6 2 2 7-7V3z"></path></g></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:copy-cell" title="Copy this cell (C)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 18 18" width="16" data-icon="ui-components:copy"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M11.9,1H3.2C2.4,1,1.7,1.7,1.7,2.5v10.2h1.5V2.5h8.7V1z M14.1,3.9h-8c-0.8,0-1.5,0.7-1.5,1.5v10.2c0,0.8,0.7,1.5,1.5,1.5h8 c0.8,0,1.5-0.7,1.5-1.5V5.4C15.5,4.6,14.9,3.9,14.1,3.9z M14.1,15.5h-8V5.4h8V15.5z"></path></g></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:paste-cell-below" title="Paste this cell from the clipboard (V)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" data-icon="ui-components:paste"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M19 2h-4.18C14.4.84 13.3 0 12 0c-1.3 0-2.4.84-2.82 2H5c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-7 0c.55 0 1 .45 1 1s-.45 1-1 1-1-.45-1-1 .45-1 1-1zm7 18H5V4h2v3h10V4h2v16z"></path></g></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:run-cell-and-select-next" title="Run this cell and advance (Shift+Enter)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" data-icon="ui-components:run"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M8 5v14l11-7z"></path></g></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:interrupt-kernel" title="Interrupt the kernel"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" height="24" viewBox="0 0 24 24" width="24" data-icon="ui-components:stop"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M0 0h24v24H0z" fill="none"></path><path d="M6 6h12v12H6z"></path></g></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:restart-kernel" title="Restart the kernel"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 18 18" data-icon="ui-components:refresh"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M9 13.5c-2.49 0-4.5-2.01-4.5-4.5S6.51 4.5 9 4.5c1.24 0 2.36.52 3.17 1.33L10 8h5V3l-1.76 1.76C12.15 3.68 10.66 3 9 3 5.69 3 3.01 5.69 3.01 9S5.69 15 9 15c2.97 0 5.43-2.16 5.9-5h-1.52c-.46 2-2.24 3.5-4.38 3.5z"></path></g></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:restart-run-all" title="Restart the kernel and run all cells"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" data-icon="ui-components:fast-forward"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><path d="M4 18l8.5-6L4 6v12zm9-12v12l8.5-6L13 6z"></path></g></svg></span></button></div><div class="lm-Widget jp-Notebook-toolbarCellType jp-Toolbar-item"><div class="jp-HTMLSelect jp-DefaultStyle jp-Notebook-toolbarCellTypeDropdown"><select aria-label="Cell type" title="Select the cell type"><option value="-">-</option><option value="code">Code</option><option value="markdown">Markdown</option><option value="raw">Raw</option></select><span class="f1ozlkqi"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 18 18" data-icon="ui-components:caret-down-empty"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161" shape-rendering="geometricPrecision"><path d="M5.2,5.9L9,9.7l3.8-3.8l1.2,1.2l-4.9,5l-4.9-5L5.2,5.9z"></path></g></svg></span></div></div><div class="lm-Widget jp-Toolbar-spacer jp-Toolbar-item"></div><div class="lm-Widget jp-CommandToolbarButton jp-nb-interface-switcher-button jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="jupyter-notebook:open-lab" title="JupyterLab"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="32" data-icon="ui-components:launch"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3 jp-icon-selectable" fill="#616161"><path d="M26,28H6a2.0027,2.0027,0,0,1-2-2V6A2.0027,2.0027,0,0,1,6,4H16V6H6V26H26V16h2V26A2.0027,2.0027,0,0,1,26,28Z"></path><polygon points="20 2 20 4 26.586 4 18 12.586 19.414 14 28 5.414 28 12 30 12 30 2 20 2"></polygon></g></svg></span><span class="jp-ToolbarButtonComponent-label">JupyterLab</span></button></div><div class="lm-Widget jp-ToolbarButton jp-Toolbar-item"><button class="jp-DebuggerBugButton jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-pressed="false" aria-disabled="true" disabled="" title="Select a kernel that supports debugging to enable debugger"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" data-icon="ui-components:bug"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3 jp-icon-selectable" fill="#616161"><path d="M20 8h-2.81c-.45-.78-1.07-1.45-1.82-1.96L17 4.41 15.59 3l-2.17 2.17C12.96 5.06 12.49 5 12 5c-.49 0-.96.06-1.41.17L8.41 3 7 4.41l1.62 1.63C7.88 6.55 7.26 7.22 6.81 8H4v2h2.09c-.05.33-.09.66-.09 1v1H4v2h2v1c0 .34.04.67.09 1H4v2h2.81c1.04 1.79 2.97 3 5.19 3s4.15-1.21 5.19-3H20v-2h-2.09c.05-.33.09-.66.09-1v-1h2v-2h-2v-1c0-.34-.04-.67-.09-1H20V8zm-6 8h-4v-2h4v2zm0-4h-4v-2h4v2z"></path></g></svg></span></button></div><div class="lm-Widget jp-KernelName jp-Toolbar-item"><button class="jp-Toolbar-kernelName jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" title="Switch kernel"><span class="jp-ToolbarButtonComponent-label">Python 3 (ipykernel)</span></button></div><div class="lm-Widget jp-mod-highlighted jp-Toolbar-item"><div class="jp-Notebook-ExecutionIndicator" title="" data-status="idle"><div class="jp-Statusbar-ProgressCircle" role="progressbar" aria-label="Kernel status" aria-valuemin="0" aria-valuemax="100" aria-valuenow="100"><svg viewBox="0 0 250 250"><circle cx="125" cy="125" r="104" stroke="var(--jp-inverse-layout-color3)" stroke-width="20" fill="none"></circle><path transform="translate(125,125) scale(.9)" d="M 0 0 v -104 A 104 104 1 0 0 -0.0000 -104.0000 z" fill="var(--jp-inverse-layout-color3)"></path></svg></div><div class="jp-Notebook-ExecutionIndicator-tooltip down "><span> Kernel status: Idle </span></div></div></div><div class="lm-Widget jp-ToolbarButton jp-Toolbar-responsive-opener jp-Toolbar-item lm-mod-hidden"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-pressed="false" aria-disabled="false" title="More commands"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" data-icon="ui-components:ellipses"><g xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#616161"><circle cx="5" cy="12" r="2"></circle><circle cx="12" cy="12" r="2"></circle><circle cx="19" cy="12" r="2"></circle></g></svg></span></button></div></div><div class="lm-Widget lm-Panel lm-BoxPanel" data-direction="top-to-bottom" data-alignment="start" style="position: absolute; contain: strict; top: 29px; left: 0px; width: 1280px; height: 0px; min-width: 0px; min-height: 0px;"></div><div class="jp-WindowedPanel-outer lm-Widget jp-Notebook jp-mod-scrollPastEnd jp-mod-showHiddenCellsButton jp-NotebookPanel-notebook jp-mod-editMode" data-jp-kernel-user="true" data-jp-undoer="true" data-jp-code-runner="true" data-jp-traversable="true" tabindex="-1" data-lm-dragscroll="true" role="region" aria-label="notebook content" id="id-903b8969-4d00-4909-aec6-6a4107d8310c" style="position: absolute; contain: strict; top: 29px; left: 0px; width: 1280px; height: 539px;"><div class="jp-WindowedPanel-inner"><div class="jp-WindowedPanel-window" style="position: relative; top: 0px;"><div class="lm-Widget jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" aria-label="Code Cell Content" data-windowed-list-index="0"><div class="lm-Widget jp-CellHeader jp-Cell-header"></div><div class="lm-Widget lm-Panel jp-Cell-inputWrapper"><div class="lm-Widget jp-Collapser jp-InputCollapser jp-Cell-inputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-InputArea jp-Cell-inputArea"><div class="lm-Widget jp-InputPrompt jp-InputArea-prompt">[91]:</div><div class="lm-Widget jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline"><div class="cm-editor ͼ1 ͼ2 ͼ4 ͼo" style=""><div aria-live="polite" style="position: fixed; top: -10000px;"><div>Selection deleted</div></div><div tabindex="-1" class="cm-scroller"><div spellcheck="false" autocorrect="off" autocapitalize="off" translate="no" contenteditable="true" class="cm-content" style="tab-size: 4;" role="textbox" aria-multiline="true" data-language="python"><div class="cm-line"><span class="ͼs">import</span> numpy <span class="ͼs">as</span> np</div><div class="cm-line"><span class="ͼs">import</span> matplotlib<span class="ͼ19">.</span>pyplot <span class="ͼs">as</span> plt</div><div class="cm-line"><span class="ͼs">from</span> scipy<span class="ͼ19">.</span>optimize <span class="ͼs">import</span> minimize</div><div class="cm-line"><br></div><div class="cm-line"><span class="ͼ11"># Glaso correlation coefficients</span></div><div class="cm-line">coefficients <span class="ͼ10">=</span> <span class="ͼ14">[</span><span class="ͼu">0.989</span><span class="ͼ19">,</span> <span class="ͼu">0.172</span><span class="ͼ19">,</span> <span class="ͼu">1.2255</span><span class="ͼ19">,</span> <span class="ͼu">2.8869</span><span class="ͼ19">,</span> <span class="ͼu">14.1811</span><span class="ͼ19">,</span> <span class="ͼu">3.3053</span><span class="ͼ19">,</span> <span class="ͼu">0.5</span><span class="ͼ14">]</span></div><div class="cm-line"><span class="ͼ11"># Objective function (GOR estimation error)</span></div><div class="cm-line"><br></div><div class="cm-line"><span class="ͼ11">#ACO parameters</span></div><div class="cm-line">alpha <span class="ͼ10">=</span> <span class="ͼu">1</span></div><div class="cm-line">beta <span class="ͼ10">=</span> <span class="ͼu">2</span></div><div class="cm-line">rho <span class="ͼ10">=</span> <span class="ͼu">0.1</span></div><div class="cm-line">Q <span class="ͼ10">=</span> <span class="ͼu">100</span></div><div class="cm-line">max_iterations <span class="ͼ10">=</span> <span class="ͼu">200</span></div><div class="cm-line"><br></div><div class="cm-line"><span class="ͼ11"># Main program</span></div><div class="cm-line"><span class="ͼs">if</span> __name__ <span class="ͼ10">==</span> <span class="ͼ12">'__main__'</span>:</div><div class="cm-line">    p <span class="ͼ10">=</span> <span class="ͼu">2405</span>  <span class="ͼ11"># pressure</span></div><div class="cm-line">    API <span class="ͼ10">=</span> <span class="ͼu">37</span>  <span class="ͼ11"># API gravity</span></div><div class="cm-line">    T <span class="ͼ10">=</span> <span class="ͼu">60</span>  <span class="ͼ11"># temperature</span></div><div class="cm-line">    actual_GOR <span class="ͼ10">=</span> <span class="ͼu">737</span>  <span class="ͼ11"># actual GOR value</span></div></div><div class="cm-layer cm-layer-above cm-cursorLayer" aria-hidden="true" style="z-index: 150; animation-duration: 1200ms; animation-name: cm-blink;"><div class="cm-cursor cm-cursor-primary" style="left: 127.531px; top: 369.292px; height: 15.3333px;"></div></div><div class="cm-layer cm-selectionLayer" aria-hidden="true" style="z-index: -2;"></div></div></div></div></div></div><div class="lm-Widget jp-CellResizeHandle"></div><div class="lm-Widget lm-Panel jp-Cell-outputWrapper"><div class="lm-Widget jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-OutputArea jp-Cell-outputArea" style=""><div class="jp-OutputArea-promptOverlay" title="Toggle output scrolling"></div></div></div><div class="lm-Widget jp-CellFooter jp-Cell-footer"></div></div><div class="lm-Widget jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" aria-label="Code Cell Content" data-windowed-list-index="1"><div class="lm-Widget jp-CellHeader jp-Cell-header"></div><div class="lm-Widget lm-Panel jp-Cell-inputWrapper"><div class="lm-Widget jp-Collapser jp-InputCollapser jp-Cell-inputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-InputArea jp-Cell-inputArea"><div class="lm-Widget jp-InputPrompt jp-InputArea-prompt">[93]:</div><div class="lm-Widget jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline"><div class="cm-editor ͼ1 ͼ2 ͼ4 ͼo" style=""><div aria-live="polite" style="position: fixed; top: -10000px;"></div><div tabindex="-1" class="cm-scroller"><div spellcheck="false" autocorrect="off" autocapitalize="off" translate="no" contenteditable="true" class="cm-content" style="tab-size: 4;" role="textbox" aria-multiline="true" data-language="python"><div class="cm-line"><span class="ͼs">def</span> <span class="ͼv">objective_function</span><span class="ͼ19">(</span>coefficients<span class="ͼ19">,</span> p<span class="ͼ19">,</span> API<span class="ͼ19">,</span> T<span class="ͼ19">,</span> actual_GOR<span class="ͼ19">)</span>:</div><div class="cm-line">    predicted_GOR <span class="ͼ10">=</span> glaso_correlation<span class="ͼ19">(</span>p<span class="ͼ19">,</span> API<span class="ͼ19">,</span> T<span class="ͼ19">,</span> coefficients<span class="ͼ19">)</span></div><div class="cm-line">    actual_gor <span class="ͼ10">=</span> <span class="ͼu">737</span>  <span class="ͼ11"># actual GOR values</span></div><div class="cm-line">    <span class="ͼs">return</span> np<span class="ͼ19">.</span><span class="ͼz">sum</span><span class="ͼ19">(</span><span class="ͼ19">(</span>predicted_GOR <span class="ͼ10">-</span> actual_GOR<span class="ͼ19">)</span> <span class="ͼ10">**</span> <span class="ͼu">2</span><span class="ͼ19">)</span></div><div class="cm-line"><br></div></div><div class="cm-layer cm-layer-above cm-cursorLayer" aria-hidden="true" style="z-index: 150; animation-duration: 1200ms; animation-name: cm-blink2;"><div class="cm-cursor cm-cursor-primary" style="left: 299.083px; top: 41.7292px; height: 15.3333px;"></div></div><div class="cm-layer cm-selectionLayer" aria-hidden="true" style="z-index: -2;"></div></div></div></div></div></div><div class="lm-Widget jp-CellResizeHandle"></div><div class="lm-Widget lm-Panel jp-Cell-outputWrapper"><div class="lm-Widget jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-OutputArea jp-Cell-outputArea"><div class="jp-OutputArea-promptOverlay" title="Toggle output scrolling"></div></div></div><div class="lm-Widget jp-CellFooter jp-Cell-footer"></div></div><div class="lm-Widget jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs jp-mod-active jp-mod-selected jp-mod-dirty" aria-label="Code Cell Content" data-windowed-list-index="2"><div class="lm-Widget jp-Toolbar jp-cell-menu jp-cell-toolbar" style="min-height: var(--jp-private-toolbar-height);"><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="assistant:cell"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 16 16" fill="none" data-icon="assistant:wizard"><path xmlns="http://www.w3.org/2000/svg" d="M10.125 3.25C10.0312 3.1875 10 3.09375 10 3C10 2.90625 10.0312 2.84375 10.125 2.78125L12 2L12.75 0.15625C12.8125 0.0625 12.9062 0 13 0C13.0938 0 13.1562 0.0625 13.2188 0.15625L14 2L15.8438 2.78125C15.9375 2.84375 16 2.90625 16 3C16 3.09375 15.9375 3.1875 15.8438 3.25L14 4L13.2188 5.875C13.1562 5.96875 13.0938 6 13 6C12.9062 6 12.8125 5.96875 12.75 5.875L12 4L10.125 3.25ZM15.8438 12.7812C15.9375 12.8438 16 12.9062 16 13C16 13.0938 15.9375 13.1875 15.8438 13.25L14 14L13.2188 15.875C13.1562 15.9688 13.0938 16 13 16C12.9062 16 12.8125 15.9688 12.75 15.875L12 14L10.125 13.25C10.0312 13.1875 10 13.0938 10 13C10 12.9062 10.0312 12.8438 10.125 12.7812L12 12L12.75 10.1562C12.8125 10.0625 12.9062 10 13 10C13.0938 10 13.1562 10.0625 13.2188 10.1562L14 12L15.8438 12.7812ZM12 8C12 8.1875 11.875 8.375 11.7188 8.4375L8.1875 10.2188L6.4375 13.75C6.34375 13.9062 6.15625 14 6 14C5.78125 14 5.625 13.9062 5.53125 13.75L3.78125 10.2188L0.25 8.4375C0.09375 8.375 0 8.1875 0 8C0 7.8125 0.09375 7.625 0.25 7.5625L3.78125 5.78125L5.53125 2.28125C5.71875 1.9375 6.25 1.9375 6.4375 2.28125L8.1875 5.78125L11.7188 7.5625C11.875 7.625 12 7.8125 12 8Z" fill="#31a824"></path></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:duplicate-below" title="Create a duplicate of this cell below"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none" data-icon="ui-components:duplicate"><path xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill-rule="evenodd" clip-rule="evenodd" d="M2.79998 0.875H8.89582C9.20061 0.875 9.44998 1.13914 9.44998 1.46198C9.44998 1.78482 9.20061 2.04896 8.89582 2.04896H3.35415C3.04936 2.04896 2.79998 2.3131 2.79998 2.63594V9.67969C2.79998 10.0025 2.55061 10.2667 2.24582 10.2667C1.94103 10.2667 1.69165 10.0025 1.69165 9.67969V2.04896C1.69165 1.40328 2.1904 0.875 2.79998 0.875ZM5.36665 11.9V4.55H11.0833V11.9H5.36665ZM4.14165 4.14167C4.14165 3.69063 4.50728 3.325 4.95832 3.325H11.4917C11.9427 3.325 12.3083 3.69063 12.3083 4.14167V12.3083C12.3083 12.7594 11.9427 13.125 11.4917 13.125H4.95832C4.50728 13.125 4.14165 12.7594 4.14165 12.3083V4.14167Z" fill="#616161"></path><path xmlns="http://www.w3.org/2000/svg" class="jp-icon3" d="M9.43574 8.26507H8.36431V9.3365C8.36431 9.45435 8.26788 9.55078 8.15002 9.55078C8.03217 9.55078 7.93574 9.45435 7.93574 9.3365V8.26507H6.86431C6.74645 8.26507 6.65002 8.16864 6.65002 8.05078C6.65002 7.93292 6.74645 7.8365 6.86431 7.8365H7.93574V6.76507C7.93574 6.64721 8.03217 6.55078 8.15002 6.55078C8.26788 6.55078 8.36431 6.64721 8.36431 6.76507V7.8365H9.43574C9.5536 7.8365 9.65002 7.93292 9.65002 8.05078C9.65002 8.16864 9.5536 8.26507 9.43574 8.26507Z" fill="#616161" stroke="#616161" stroke-width="0.5"></path></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:move-cell-up" title="Move this cell up (Ctrl+Shift+Up)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none" data-icon="ui-components:move-up"><path xmlns="http://www.w3.org/2000/svg" class="jp-icon3" d="M1.52899 6.47101C1.23684 6.76316 1.23684 7.23684 1.52899 7.52899V7.52899C1.82095 7.82095 2.29426 7.82116 2.58649 7.52946L6.25 3.8725V12.25C6.25 12.6642 6.58579 13 7 13V13C7.41421 13 7.75 12.6642 7.75 12.25V3.8725L11.4027 7.53178C11.6966 7.82619 12.1736 7.82641 12.4677 7.53226V7.53226C12.7617 7.2383 12.7617 6.7617 12.4677 6.46774L7.70711 1.70711C7.31658 1.31658 6.68342 1.31658 6.29289 1.70711L1.52899 6.47101Z" fill="#616161"></path></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:move-cell-down" title="Move this cell down (Ctrl+Shift+Down)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none" data-icon="ui-components:move-down"><path xmlns="http://www.w3.org/2000/svg" class="jp-icon3" d="M12.471 7.52899C12.7632 7.23684 12.7632 6.76316 12.471 6.47101V6.47101C12.179 6.17905 11.7057 6.17884 11.4135 6.47054L7.75 10.1275V1.75C7.75 1.33579 7.41421 1 7 1V1C6.58579 1 6.25 1.33579 6.25 1.75V10.1275L2.59726 6.46822C2.30338 6.17381 1.82641 6.17359 1.53226 6.46774V6.46774C1.2383 6.7617 1.2383 7.2383 1.53226 7.53226L6.29289 12.2929C6.68342 12.6834 7.31658 12.6834 7.70711 12.2929L12.471 7.52899Z" fill="#616161"></path></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:insert-cell-above" title="Insert a cell above (A)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none" data-icon="ui-components:add-above"><g xmlns="http://www.w3.org/2000/svg" clip-path="url(#clip0_137_19492)"><path class="jp-icon3" d="M4.75 4.93066H6.625V6.80566C6.625 7.01191 6.79375 7.18066 7 7.18066C7.20625 7.18066 7.375 7.01191 7.375 6.80566V4.93066H9.25C9.45625 4.93066 9.625 4.76191 9.625 4.55566C9.625 4.34941 9.45625 4.18066 9.25 4.18066H7.375V2.30566C7.375 2.09941 7.20625 1.93066 7 1.93066C6.79375 1.93066 6.625 2.09941 6.625 2.30566V4.18066H4.75C4.54375 4.18066 4.375 4.34941 4.375 4.55566C4.375 4.76191 4.54375 4.93066 4.75 4.93066Z" fill="#616161" stroke="#616161" stroke-width="0.7"></path></g><path xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill-rule="evenodd" clip-rule="evenodd" d="M11.5 9.5V11.5L2.5 11.5V9.5L11.5 9.5ZM12 8C12.5523 8 13 8.44772 13 9V12C13 12.5523 12.5523 13 12 13L2 13C1.44772 13 1 12.5523 1 12V9C1 8.44772 1.44771 8 2 8L12 8Z" fill="#616161"></path><defs xmlns="http://www.w3.org/2000/svg"><clippath id="clip0_137_19492"><rect class="jp-icon3" width="6" height="6" fill="white" transform="matrix(-1 0 0 1 10 1.55566)"></rect></clippath></defs></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:insert-cell-below" title="Insert a cell below (B)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 14 14" fill="none" data-icon="ui-components:add-below"><g xmlns="http://www.w3.org/2000/svg" clip-path="url(#clip0_137_19498)"><path class="jp-icon3" d="M9.25 10.0693L7.375 10.0693L7.375 8.19434C7.375 7.98809 7.20625 7.81934 7 7.81934C6.79375 7.81934 6.625 7.98809 6.625 8.19434L6.625 10.0693L4.75 10.0693C4.54375 10.0693 4.375 10.2381 4.375 10.4443C4.375 10.6506 4.54375 10.8193 4.75 10.8193L6.625 10.8193L6.625 12.6943C6.625 12.9006 6.79375 13.0693 7 13.0693C7.20625 13.0693 7.375 12.9006 7.375 12.6943L7.375 10.8193L9.25 10.8193C9.45625 10.8193 9.625 10.6506 9.625 10.4443C9.625 10.2381 9.45625 10.0693 9.25 10.0693Z" fill="#616161" stroke="#616161" stroke-width="0.7"></path></g><path xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill-rule="evenodd" clip-rule="evenodd" d="M2.5 5.5L2.5 3.5L11.5 3.5L11.5 5.5L2.5 5.5ZM2 7C1.44772 7 1 6.55228 1 6L1 3C1 2.44772 1.44772 2 2 2L12 2C12.5523 2 13 2.44772 13 3L13 6C13 6.55229 12.5523 7 12 7L2 7Z" fill="#616161"></path><defs xmlns="http://www.w3.org/2000/svg"><clippath id="clip0_137_19498"><rect class="jp-icon3" width="6" height="6" fill="white" transform="matrix(1 1.74846e-07 1.74846e-07 -1 4 13.4443)"></rect></clippath></defs></svg></span></button></div><div class="lm-Widget jp-CommandToolbarButton jp-Toolbar-item"><button class="jp-ToolbarButtonComponent jp-mod-minimal jp-Button" aria-disabled="false" data-command="notebook:delete-cell" title="Delete this cell (D, D)"><span class="jp-ToolbarButtonComponent-icon f1vya9e0"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16px" height="16px" data-icon="ui-components:delete"><path xmlns="http://www.w3.org/2000/svg" d="M0 0h24v24H0z" fill="none"></path><path xmlns="http://www.w3.org/2000/svg" class="jp-icon3" fill="#626262" d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"></path></svg></span></button></div></div><div class="lm-Widget jp-CellHeader jp-Cell-header"></div><div class="lm-Widget lm-Panel jp-Cell-inputWrapper"><div class="lm-Widget jp-Collapser jp-InputCollapser jp-Cell-inputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-InputArea jp-Cell-inputArea"><div class="lm-Widget jp-InputPrompt jp-InputArea-prompt">[95]:</div><div class="lm-Widget jp-CodeMirrorEditor jp-Editor jp-InputArea-editor jp-mod-focused" data-type="inline"><div class="cm-editor cm-focused ͼ1 ͼ2 ͼ4 ͼo" style=""><div aria-live="polite" style="position: fixed; top: -10000px;"></div><div tabindex="-1" class="cm-scroller"><div spellcheck="false" autocorrect="off" autocapitalize="off" translate="no" contenteditable="true" class="cm-content" style="tab-size: 4;cursor: crosshair" role="textbox" aria-multiline="true" data-language="python"><div class="cm-line"><span class="ͼ11"># Glaso correlation function</span></div><div class="cm-line"><span class="ͼs">def</span> <span class="ͼv">glaso_correlation</span><span class="ͼ19">(</span>p<span class="ͼ19">,</span> API<span class="ͼ19">,</span> T<span class="ͼ19">,</span> coefficients<span class="ͼ19">)</span>:</div><div class="cm-line">    <span class="ͼs">return</span> <span class="ͼ19">(</span>coefficients<span class="ͼ14">[</span><span class="ͼu">0</span><span class="ͼ14">]</span> <span class="ͼ10">*</span> <span class="ͼ19">(</span>p <span class="ͼ10">*</span> <span class="ͼu">10</span><span class="ͼ10">**</span><span class="ͼ19">(</span><span class="ͼ10">-</span>coefficients<span class="ͼ14">[</span><span class="ͼu">1</span><span class="ͼ14">]</span> <span class="ͼ10">*</span> API<span class="ͼ19">)</span><span class="ͼ19">)</span> <span class="ͼ10">*</span> <span class="ͼ19">(</span>T<span class="ͼ10">**</span><span class="ͼ19">(</span><span class="ͼ10">-</span>coefficients<span class="ͼ14">[</span><span class="ͼu">2</span><span class="ͼ14">]</span><span class="ͼ19">)</span><span class="ͼ19">)</span> <span class="ͼ10">*</span> <span class="ͼ19">(</span>p<span class="ͼ10">**</span>coefficients<span class="ͼ14">[</span><span class="ͼu">3</span><span class="ͼ14">]</span><span class="ͼ19">)</span><span class="ͼ19">)</span> <span class="ͼ10">+</span> coefficients<span class="ͼ14">[</span><span class="ͼu">4</span><span class="ͼ14">]</span></div><div class="cm-line"><br></div><div class="cm-line"><br></div><div class="cm-line">γg <span class="ͼ14">[</span><span class="ͼ19">(</span>〖API〗<span class="ͼ10">^</span><span class="ͼu">0.989</span><span class="ͼ10">/</span>〖<span class="ͼ19">(</span>T<span class="ͼ10">-</span><span class="ͼu">460</span><span class="ͼ19">)</span>〗<span class="ͼ10">^</span><span class="ͼu">0.172</span> <span class="ͼ19">)</span> 10x<span class="ͼ14">]</span><span class="ͼu">1.2255</span>                                                                                                   		     …………………<span class="ͼ19">.</span> <span class="ͼ19">(</span><span class="ͼu">1</span><span class="ͼ19">)</span> </div><div class="cm-line">x <span class="ͼ10">=</span> <span class="ͼu">2.8869</span> – <span class="ͼ19">(</span><span class="ͼu">14.1811</span> – 3<span class="ͼ19">.</span>3053<span class="ͼz">logP</span><span class="ͼ19">)</span><span class="ͼu">0.5</span>                                                         </div><div class="cm-line"><br></div><div class="cm-line"><br></div><div class="cm-line"><br></div></div><div class="cm-layer cm-layer-above cm-cursorLayer" aria-hidden="true" style="z-index: 150; animation-duration: 1200ms; animation-name: cm-blink2;"><div class="cm-cursor cm-cursor-primary" style="left: 6px; top: 132.719px; height: 15.3333px;"></div></div><div class="cm-layer cm-selectionLayer" aria-hidden="true" style="z-index: -2;"></div></div></div></div></div></div><div class="lm-Widget jp-CellResizeHandle"></div><div class="lm-Widget lm-Panel jp-Cell-outputWrapper"><div class="lm-Widget jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-OutputArea jp-Cell-outputArea"><div class="jp-OutputArea-promptOverlay" title="Toggle output scrolling"></div></div></div><div class="lm-Widget jp-CellFooter jp-Cell-footer"></div></div><div class="lm-Widget jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" aria-label="Code Cell Content" data-windowed-list-index="3"><div class="lm-Widget jp-CellHeader jp-Cell-header"></div><div class="lm-Widget lm-Panel jp-Cell-inputWrapper"><div class="lm-Widget jp-Collapser jp-InputCollapser jp-Cell-inputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-InputArea jp-Cell-inputArea"><div class="lm-Widget jp-InputPrompt jp-InputArea-prompt">[97]:</div><div class="lm-Widget jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline"><div class="cm-editor ͼ1 ͼ2 ͼ4 ͼo" style=""><div aria-live="polite" style="position: fixed; top: -10000px;"></div><div tabindex="-1" class="cm-scroller"><div spellcheck="false" autocorrect="off" autocapitalize="off" translate="no" contenteditable="true" class="cm-content" style="tab-size: 4;" role="textbox" aria-multiline="true" data-language="python"><div class="cm-line"><span class="ͼ11"># ACO algorithm</span></div><div class="cm-line"><span class="ͼs">def</span> <span class="ͼv">aco_algorithm</span><span class="ͼ19">(</span>p<span class="ͼ19">,</span> API<span class="ͼ19">,</span> T<span class="ͼ19">,</span> actual_GOR<span class="ͼ19">)</span>:</div><div class="cm-line">    <span class="ͼ11"># Initialize pheromone trails and parameters</span></div><div class="cm-line">    tau <span class="ͼ10">=</span> np<span class="ͼ19">.</span><span class="ͼz">ones</span><span class="ͼ19">(</span><span class="ͼ19">(</span>len<span class="ͼ19">(</span>coefficients<span class="ͼ19">)</span><span class="ͼ19">,</span> len<span class="ͼ19">(</span>coefficients<span class="ͼ19">)</span><span class="ͼ19">)</span><span class="ͼ19">)</span></div><div class="cm-line">    best_coefficients <span class="ͼ10">=</span> coefficients</div><div class="cm-line">    best_error <span class="ͼ10">=</span> float<span class="ͼ19">(</span><span class="ͼ12">'inf'</span><span class="ͼ19">)</span></div><div class="cm-line"><br></div><div class="cm-line">    <span class="ͼs">for</span> _ <span class="ͼs">in</span> range<span class="ͼ19">(</span>max_iterations<span class="ͼ19">)</span>:</div><div class="cm-line">        ants <span class="ͼ10">=</span> np<span class="ͼ19">.</span><span class="ͼz">random</span><span class="ͼ19">.</span><span class="ͼz">uniform</span><span class="ͼ19">(</span><span class="ͼu">0</span><span class="ͼ19">,</span> <span class="ͼu">1</span><span class="ͼ19">,</span> size<span class="ͼ10">=</span><span class="ͼ19">(</span>Q<span class="ͼ19">,</span> len<span class="ͼ19">(</span>coefficients<span class="ͼ19">)</span><span class="ͼ19">)</span><span class="ͼ19">)</span></div><div class="cm-line">        <span class="ͼs">for</span> ant <span class="ͼs">in</span> ants:</div><div class="cm-line">            error <span class="ͼ10">=</span> objective_function<span class="ͼ19">(</span>ant<span class="ͼ19">,</span> p<span class="ͼ19">,</span> API<span class="ͼ19">,</span> T<span class="ͼ19">,</span> actual_GOR<span class="ͼ19">)</span></div><div class="cm-line">            <span class="ͼs">if</span> error <span class="ͼ10">&lt;</span> best_error:</div><div class="cm-line">                best_error <span class="ͼ10">=</span> error</div><div class="cm-line">                best_coefficients <span class="ͼ10">=</span> ant</div><div class="cm-line">                tau <span class="ͼ10">+=</span> alpha <span class="ͼ10">*</span> error <span class="ͼ10">*</span> np<span class="ͼ19">.</span><span class="ͼz">outer</span><span class="ͼ19">(</span>ant<span class="ͼ19">,</span> ant<span class="ͼ19">)</span></div><div class="cm-line">                tau <span class="ͼ10">*=</span> <span class="ͼ19">(</span><span class="ͼu">1</span> <span class="ͼ10">-</span> rho<span class="ͼ19">)</span></div><div class="cm-line">                <span class="ͼs">return</span> best_coefficients</div></div><div class="cm-layer cm-layer-above cm-cursorLayer" aria-hidden="true" style="z-index: 150; animation-duration: 1200ms; animation-name: cm-blink;"><div class="cm-cursor cm-cursor-primary" style="left: 263.333px; top: 78.125px; height: 15.3333px;"></div></div><div class="cm-layer cm-selectionLayer" aria-hidden="true" style="z-index: -2;"></div></div></div></div></div></div><div class="lm-Widget jp-CellResizeHandle"></div><div class="lm-Widget lm-Panel jp-Cell-outputWrapper"><div class="lm-Widget jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-OutputArea jp-Cell-outputArea" style=""><div class="jp-OutputArea-promptOverlay" title="Toggle output scrolling"></div></div></div><div class="lm-Widget jp-CellFooter jp-Cell-footer"></div></div><div class="lm-Widget jp-Cell jp-CodeCell jp-Notebook-cell" aria-label="Code Cell Content with Output" data-windowed-list-index="4"><div class="lm-Widget jp-CellHeader jp-Cell-header"></div><div class="lm-Widget lm-Panel jp-Cell-inputWrapper"><div class="lm-Widget jp-Collapser jp-InputCollapser jp-Cell-inputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-InputArea jp-Cell-inputArea"><div class="lm-Widget jp-InputPrompt jp-InputArea-prompt">[99]:</div><div class="lm-Widget jp-CodeMirrorEditor jp-Editor jp-InputArea-editor jp-mod-in-leading-whitespace" data-type="inline"><div class="cm-editor ͼ1 ͼ2 ͼ4 ͼo" style=""><div aria-live="polite" style="position: fixed; top: -10000px;"></div><div tabindex="-1" class="cm-scroller"><div spellcheck="false" autocorrect="off" autocapitalize="off" translate="no" contenteditable="true" class="cm-content" style="tab-size: 4;" role="textbox" aria-multiline="true" data-language="python"><div class="cm-line">optimized_coefficients <span class="ͼ10">=</span> aco_algorithm<span class="ͼ19">(</span>p<span class="ͼ19">,</span> API<span class="ͼ19">,</span> T<span class="ͼ19">,</span> actual_GOR<span class="ͼ19">)</span></div><div class="cm-line">predicted_GOR <span class="ͼ10">=</span> glaso_correlation<span class="ͼ19">(</span>p<span class="ͼ19">,</span> API<span class="ͼ19">,</span> T<span class="ͼ19">,</span> optimized_coefficients<span class="ͼ19">)</span></div><div class="cm-line">    </div><div class="cm-line">print<span class="ͼ19">(</span><span class="ͼ12">'Optimized coefficients:'</span><span class="ͼ19">,</span> optimized_coefficients<span class="ͼ19">)</span></div><div class="cm-line">print<span class="ͼ19">(</span><span class="ͼ12">'Predicted GOR:'</span><span class="ͼ19">,</span> predicted_GOR<span class="ͼ19">)</span></div></div><div class="cm-layer cm-layer-above cm-cursorLayer" aria-hidden="true" style="z-index: 150; animation-duration: 1200ms; animation-name: cm-blink;"><div class="cm-cursor cm-cursor-primary" style="left: 34.5938px; top: 41.7292px; height: 15.3333px;"></div></div><div class="cm-layer cm-selectionLayer" aria-hidden="true" style="z-index: -2;"></div></div></div></div></div></div><div class="lm-Widget jp-CellResizeHandle"></div><div class="lm-Widget lm-Panel jp-Cell-outputWrapper"><div class="lm-Widget jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-OutputArea jp-Cell-outputArea" style=""><div class="lm-Widget lm-Panel jp-OutputArea-child"><div class="lm-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget jp-RenderedText jp-mod-trusted jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stdout"><pre>Optimized coefficients: [0.17413702 0.09094106 0.95282353 0.31880656 0.10090817 0.97671059
 0.09228165]
Predicted GOR: 0.14464471224601239
</pre></div></div><div class="jp-OutputArea-promptOverlay" title="Toggle output scrolling"></div></div></div><div class="lm-Widget jp-CellFooter jp-Cell-footer"></div></div><div class="lm-Widget jp-Cell jp-CodeCell jp-Notebook-cell" aria-label="Code Cell Content with Output" data-windowed-list-index="5"><div class="lm-Widget jp-CellHeader jp-Cell-header"></div><div class="lm-Widget lm-Panel jp-Cell-inputWrapper"><div class="lm-Widget jp-Collapser jp-InputCollapser jp-Cell-inputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-InputArea jp-Cell-inputArea"><div class="lm-Widget jp-InputPrompt jp-InputArea-prompt">[85]:</div><div class="lm-Widget jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline"><div class="cm-editor ͼ1 ͼ2 ͼ4 ͼo" style=""><div aria-live="polite" style="position: fixed; top: -10000px;"></div><div tabindex="-1" class="cm-scroller"><div spellcheck="false" autocorrect="off" autocapitalize="off" translate="no" contenteditable="true" class="cm-content" style="tab-size: 4;" role="textbox" aria-multiline="true" data-language="python"><div class="cm-line"><span class="ͼ11"># Plot convergence graph</span></div><div class="cm-line">iterations <span class="ͼ10">=</span> np<span class="ͼ19">.</span><span class="ͼz">arange</span><span class="ͼ19">(</span><span class="ͼu">1</span><span class="ͼ19">,</span> max_iterations <span class="ͼ10">+</span> <span class="ͼu">1</span><span class="ͼ19">)</span></div><div class="cm-line">objective_values <span class="ͼ10">=</span> <span class="ͼ14">[</span>objective_function<span class="ͼ19">(</span>aco_algorithm<span class="ͼ19">(</span>p<span class="ͼ19">,</span> API<span class="ͼ19">,</span> T<span class="ͼ19">,</span> actual_GOR<span class="ͼ19">)</span><span class="ͼ19">,</span> p<span class="ͼ19">,</span> API<span class="ͼ19">,</span> T<span class="ͼ19">,</span> actual_GOR<span class="ͼ19">)</span> <span class="ͼs">for</span> _ <span class="ͼs">in</span> iterations<span class="ͼ14">]</span></div><div class="cm-line">plt<span class="ͼ19">.</span><span class="ͼz">plot</span><span class="ͼ19">(</span>iterations<span class="ͼ19">,</span> objective_values<span class="ͼ19">)</span></div><div class="cm-line">plt<span class="ͼ19">.</span><span class="ͼz">xlabel</span><span class="ͼ19">(</span><span class="ͼ12">'Iterations'</span><span class="ͼ19">)</span></div><div class="cm-line">plt<span class="ͼ19">.</span><span class="ͼz">ylabel</span><span class="ͼ19">(</span><span class="ͼ12">'Objective Function Value'</span><span class="ͼ19">)</span></div><div class="cm-line">plt<span class="ͼ19">.</span><span class="ͼz">title</span><span class="ͼ19">(</span><span class="ͼ12">'ACO Convergence Graph'</span><span class="ͼ19">)</span></div><div class="cm-line">plt<span class="ͼ19">.</span><span class="ͼz">show</span><span class="ͼ19">(</span><span class="ͼ19">)</span></div><div class="cm-line"><br></div><div class="cm-line"><br></div></div><div class="cm-layer cm-layer-above cm-cursorLayer" aria-hidden="true" style="z-index: 150; animation-duration: 1200ms; animation-name: cm-blink;"><div class="cm-cursor cm-cursor-primary" style="left: 6px; top: 150.917px; height: 15.3333px;"></div></div><div class="cm-layer cm-selectionLayer" aria-hidden="true" style="z-index: -2;"></div></div></div></div></div></div><div class="lm-Widget jp-CellResizeHandle"></div><div class="lm-Widget lm-Panel jp-Cell-outputWrapper"><div class="lm-Widget jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-OutputArea jp-Cell-outputArea" style=""><div class="lm-Widget lm-Panel jp-OutputArea-child"><div class="lm-Widget jp-OutputPrompt jp-OutputArea-prompt"></div><div class="lm-Widget jp-RenderedImage jp-mod-trusted jp-OutputArea-output" data-mime-type="image/png"><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAioAAAHFCAYAAADcytJ5AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8fJSN1AAAACXBIWXMAAA9hAAAPYQGoP6dpAABKE0lEQVR4nO3dd3hUZf7+8XtII4QQSiAhECCCkd4VwVWKwBIF5WdZUFaDdUEQFPyqWEFFUCyoKKirgK4KuwhY6d2CgoAgIFICQSFAaKFoSDLP7w+YMZMJMGeSyRTfr+vKRebMyczncBJy85zPeR6bMcYIAAAgAJXzdwEAAABnQ1ABAAABi6ACAAACFkEFAAAELIIKAAAIWAQVAAAQsAgqAAAgYBFUAABAwCKoAACAgEVQAYp49dVXZbPZ1LRp03Put2PHDg0ePFipqamKjo5WhQoV1KRJEz322GP67bffXPY1xujDDz9Uly5dVKVKFUVFRemCCy7QoEGDtHv3bkv1rV+/XrfddptSUlJUvnx5VaxYUa1bt9bzzz+vQ4cOWT5e+M9XX32lm266SXXq1FFUVJRiYmLUpEkTDR8+XD///LNfa7PZbBo8eLBfawAkycYU+oCrli1b6scff5QkrVy5Uu3atXPb5/PPP1ffvn0VHx+vwYMHq1WrVrLZbNqwYYPeffddlStXTmvXrpUk2e123XzzzZo+fbpuuukm9enTR3FxcVq/fr3GjRun48eP6/PPP9dll1123trefvtt3XPPPbrooot0zz33qHHjxsrLy9Pq1av19ttvq0WLFpo1a1bp/oXAJx577DGNHj1a7du3V//+/XXhhRcqPz9f69ev19SpU7Vhwwbl5+crLCzML/XZbDYNGjRIEyZM8Mv7A04GgNOqVauMJHP11VcbSeauu+5y22fHjh0mJibGtGrVyhw5csTtebvdbj7++GPn42effdZIMmPHjnXbNysry9StW9ckJCSYw4cPn7O2b775xoSFhZkePXqYP/74w+353Nxc88knn3hwlIErPz+/2GMLNR9++KGRZAYMGGDsdrvb83a73UyYMMHk5+ef83VOnDjhqxKNJDNo0CCfvT7gKYIKUMiAAQOMJLNhwwbToUMHExsb6/bLYPDgwUaS+fbbb8/7erm5uaZKlSqmUaNGxf5CMubPX1ovvPDCOV+rZ8+eJjw83GRmZnp0LAUFBea5554zF110kYmMjDTVq1c3t9xyi9m9e7fLfh07djRNmjQx33//vfnb3/5moqOjTUpKihkzZowpKCgwxhizf/9+ExERYR577DG399m8ebORZF555RXntr1795q7777b1KpVy0RERJh69eqZkSNHmry8POc+GRkZRpJ57rnnzNNPP23q1atnwsLCzJw5c4wxxsyePds0a9bMREZGmpSUFDN+/Hjz5JNPmqL/v7Lb7eb11183LVq0MOXLlzeVK1c2119/vdm+fbvl43Q4fPiwGTZsmElJSXH+3aWlpZnNmzc798nNzTVPP/208+83Pj7e9O/f3+zfv/+856Zx48YmPj7e/P777+fdt2j9y5YtM+3btzfR0dGmT58+xhhjpk2bZrp162YSExNN+fLlTcOGDc1DDz1kjh8/7vIa6enpJiYmxvz000+mS5cupkKFCiY+Pt4MGjTI7fvcEVTee+8907BhQxMdHW2aN29uPvvsM49rBkoDQQU44+TJkyYuLs5cfPHFxhhj/v3vfxtJZsqUKS77paammoSEBI9e85tvvjGSzEMPPXTWfY4dO2bKlStn/v73v591n/z8fFOhQgXTrl07j97XGGPuvvtuI8kMHjzYzJ0710yaNMlUr17dJCcnmwMHDjj369ixo6lWrZq58MILzaRJk8yCBQvMPffcYySZqVOnOvf7f//v/5nk5GS3X+oPPvigiYyMNNnZ2caY0yElOTnZ1K1b17z55ptm4cKF5umnnzZRUVGmf//+zq9zBJVatWqZzp07mxkzZpj58+ebjIwMM2fOHFOuXDnTqVMnM2vWLPO///3PtGvXztSrV88tqNx1110mIiLCDB8+3MydO9d8+OGHpmHDhiYhIcFkZWVZPs6cnBzTpEkTExMTY5566ikzb9488/HHH5uhQ4eaxYsXG2NOh8AePXqYmJgYM2rUKLNgwQLz73//29SqVcs0btzYnDx58qzn5bfffjOSzE033eTxuXTUX7VqVZOcnGxee+01s2TJErNs2TJjjDFPP/20efnll80XX3xhli5daiZNmmRSUlJM586dXV4jPT3dREZGmjp16pjRo0eb+fPnm5EjR5rw8HDTs2dPl30lmXr16plLLrnE/Pe//zVffvml6dSpkwkPD3cLgYAvEVSAM9577z0jyUyaNMkYczpAVKxY0Vx++eUu+5UvX95ceumlHr3mtGnTXF7zbBISEkyjRo3O+nxWVpaRZPr27evR+zpGOe655x6X7d99952RZB555BHnto4dOxpJ5rvvvnPZt3Hjxi7h6dNPPzWSzPz5853b8vPzTVJSkrn++uud2/71r3+ZihUrml27drm83gsvvGAkmY0bNxpj/gwq9evXN6dOnXLZ9+KLLzbJyckmNzfXue3YsWOmWrVqLkHl22+/NZLMiy++6PL1u3fvNtHR0ebBBx+0fJxPPfWUkWQWLFhgzuajjz4yklwu8Rnz56XDN95446xfu3LlSiPJPPzww27P5efnm7y8POdH4VE4R/2LFi0662sbc3qEKS8vzyxbtsxIMj/++KPzufT0dLfRL2OMGT16tJFkvvrqK+c2SSYhIcHk5OQ4t2VlZZly5cqZMWPGnLMGoDSFzF0/y5cvV69evZSUlCSbzabZs2dbfo3//ve/atmypSpUqKC6detq3LhxpV8oAtY777yj6Oho9e3bV5JUsWJF3XjjjVqxYoW2bt3q0/c2xshms5Xa6y1ZskSS1L9/f5ftl1xyiRo1aqRFixa5bE9MTNQll1zisq158+batWuX83FaWpoSExM1efJk57Z58+Zpz549uv32253bPv/8c3Xu3FlJSUnKz893fqSlpUmSli1b5vI+11xzjSIiIpyPT5w4odWrV6t3796KjIx0bq9YsaJ69erl8rWff/65bDab/vnPf7q8V2Jiolq0aKGlS5daPs45c+YoNTVVXbt21dl8/vnnqly5snr16uXyvi1btlRiYqLb+3qqWrVqioiIcH58/PHHLs9XqVJFXbp0cfu6HTt26Oabb1ZiYqLCwsIUERGhjh07SpI2b97stn+/fv1cHt98882S/vy+cejcubNiY2OdjxMSElSjRg2Xvy/A10ImqJw4cUItWrTwukN9zpw56tevnwYMGKCffvpJb7zxhl566SU63v8itm3bpuXLl+vqq6+WMUZHjhzRkSNHdMMNN0iS3n33Xee+derUUUZGhkevW6dOHUk65/4nTpxQdna2kpOTz7pPfHy8KlSo4PH7Hjx4UJJUs2ZNt+eSkpKczztUq1bNbb+oqCj9/vvvzsfh4eG65ZZbNGvWLB05ckSSNGXKFNWsWVN///vfnfvt27dPn332mcsv3IiICDVp0kSSlJ2d7fI+RWs8fPiwjDFKSEhwq6notn379jn3Lfp+K1eudHsvT47zwIEDql27ttt+Rd/3yJEjioyMdHvfrKwst/ctzHGei/tlv3TpUq1atUqTJk0q9muLO5/Hjx/X5Zdfru+++07PPPOM8zVmzpwpSS7HJp0+j0X/HhITEyXJq+8LwNfC/V1AaUlLS3P+j604p06d0mOPPaYPPvhAR44cUdOmTfXcc8+pU6dOkqT3339fvXv31oABAyRJF1xwgR566CE999xzGjRoUKn+bxeB591335UxRjNmzNCMGTPcnp86daqeeeYZhYWF6e9//7tee+01rVy5Updeeuk5X7dNmzaqUqWKPv30U40ZM6bY76NPP/1Udrtd3bp1O+vrhIWF6corr9ScOXP066+/nvcXqeMXzN69e9323bNnj+Lj48/59Wdz2223ady4cZo2bZr69OmjTz/9VPfdd5/LLbTx8fFq3ry5Ro8eXexrJCUluTwu+ndSpUoV2Ww27du3z+1rs7KyXB7Hx8fLZrNpxYoVioqKctu/uG3nU716df3666/n3Cc+Pl7VqlXT3Llzi32+8ChEUUlJSWrSpIkWLFigP/74Q+XLl3c+17JlS0mnw0dxivv+Wbx4sfbs2aOlS5c6R1EkOcNkUfn5+Tp48KBLCHH8vRYXTAB/C5kRlfO57bbb9PXXX2vatGlav369brzxRvXo0cM5pJ+bm+vyD4YkRUdH69dff2WYM8QVFBRo6tSpql+/vpYsWeL2MXz4cO3du1dz5syRJN1///2KiYnRPffco6NHj7q9njHGOZdJZGSk/u///k+bN28u9lLi/v37NWLECCUkJOjOO+88Z50jRoyQMUZ33XWXTp065fZ8Xl6ePvvsM0lyXh74z3/+47LPqlWrtHnzZl155ZUe/M24a9Sokdq1a6fJkyfrww8/VG5urm677TaXfXr27KmffvpJ9evXV9u2bd0+igaVomJiYtS2bVvNnj3b5Tgd880UfS9jjH777bdi36tZs2aWjzEtLU2//PKLFi9efNZ9evbsqYMHD6qgoKDY973ooovO+R6PPvqosrOzNWzYMJkSTmXlCC9FQ9mbb7551q/54IMPXB5/+OGHkuT8jxsQUPzWHeNDksysWbOcj7dt22ZsNpv57bffXPa78sorzYgRI4wxxrz55pumQoUKZuHChaagoMBs2bLFNGzY0Egy33zzTVmWjzL22WefOW+TLc6BAwdMVFSU6d27t8vXVKhQwdSrV8+88MILZtGiRWbRokXmtddeM61atTItW7Z07ltQUGD69OljJJmbb77ZfPLJJ2bp0qXm1VdfNcnJyaZy5couTYzn8tZbb5nw8HDTtGlT8/rrr5ulS5eaBQsWmOeff940aNDApca7777b2Gw2c99995l58+aZN99809SoUcMkJyc779Ax5s/bXotKT083devWddv+5ptvGkmmdu3apkOHDm7P79mzx9StW9c0bNjQvPHGG2bRokXmiy++MK+//rq5+uqrnbdHO5ppx40b5/YaRe/6mTFjhmnXrp2pW7eusdlsLvvefffdpkKFCub//u//zGeffWYWL15sPvjgAzNw4ECXplZPj9Nx10/FihXNM888Y+bPn28++eQTM2zYMOddP/n5+SYtLc1UrVrVjBo1ysyZM8csXLjQTJkyxaSnp5uZM2e6vU9Rjz76qJFkOnToYN566y2zZMkSs2jRIjNlyhRz5ZVXGklm7ty5560/OzvbVKlSxbRo0cLMnDnTfPbZZ6Zv377mwgsvNJLM5MmTXY71bHf9pKWlubyuzjKPSt26dU16evp5jw8oLX+JoPLf//7XSDIxMTEuH+Hh4eYf//iHMeZ0p/yDDz5oypcvb8LCwkyVKlXMyJEji71LAKGld+/eJjIy8pzzX/Tt29eEh4e73O66fft2c88995gGDRqYqKgoEx0dbRo3bmyGDRtmMjIyXL7ebrebDz74wHTq1MlUrlzZOTfIwIED3e6OOZ9169aZ9PR0U6dOHRMZGemcfO6JJ55wOQbHPCqpqakmIiLCxMfHm3/+859nnUelqLMFlaNHj5ro6Ggjybz99tvF1njgwAEzZMgQk5KSYiIiIkzVqlVNmzZtzKOPPuqc2+NcQcUYY2bNmuWcR6VOnTpm7NixZsiQIaZKlSpu+7777rumXbt2JiYmxkRHR5v69eubW2+91axevdqr4zx8+LAZOnSoqVOnjomIiDA1atQwV199tfn555+d++Tl5ZkXXnjBOX9LxYoVTcOGDc2//vUvs3Xr1mKPqajly5ebPn36mNq1a5uIiAhToUIF07hxYzNw4ECX2s9VvzGnb4Nv3769qVChgqlevbq58847zZo1a4oNKjExMWb9+vWmU6dOJjo62lStWtUMHDjQbc4VggoCRUhOoW+z2TRr1iz17t1bkjR9+nT169dPGzdudJuOumLFis5GMun0ZYCsrCxVr15dixYt0lVXXaV9+/apRo0aZXkIAIrIy8tTy5YtVatWLc2fP9/f5QSl/v37a8aMGWftgQECUcg0055Lq1atVFBQoP379+vyyy8/575hYWGqVauWJOmjjz5S+/btCSmAH9xxxx3q1q2batasqaysLE2aNEmbN2/WK6+84u/SAJShkAkqx48f17Zt25yPMzIytG7dOlWtWlWpqanq16+fbr31Vr344otq1aqVsrOztXjxYjVr1kxXXXWVsrOzNWPGDHXq1El//PGHJk+erP/9739ucz4AKBvHjh3TAw88oAMHDigiIkKtW7fWl19+ec75TQCEnpC59LN06VJ17tzZbXt6erqmTJmivLw8PfPMM3rvvff022+/qVq1amrfvr1GjRqlZs2aKTs7W7169dKGDRtkjFH79u01evToYlfOBQAAZSNkggoAAAg9f5l5VAAAQPAhqAAAgIAV1M20drtde/bsUWxsLFPcAwAQJIwxOnbsmJKSklSu3LnHTII6qOzZs+ecC7kBAIDAtXv37vOuXRbUQcWx8Nfu3btVqVIlP1cDAAA8kZOTo+Tk5HMu4OkQ1EHFcbmnUqVKBBUAAIKMJ20bNNMCAICARVABAAABi6ACAAACFkEFAAAELIIKAAAIWAQVAAAQsAgqAAAgYBFUAABAwCKoAACAgEVQAQAAAYugAgAAAhZBBQAABCyCCoAy8fupAn+XACAIEVQA+NzPWTlq8dR8PT/3Z3+XAiDIEFQA+NzmvTk6lW/Xj78e8XcpAIIMQQWAzxnj+icAeIqgAsDn7MbxJ0kFgDUEFQA+5wgodnIKAIsIKgB8zpwJKoYRFQAWEVQA+Bw9KgC8RVAB4HP0qADwFkEFgM/RowLAWwQVAD5nivwJAJ4iqADwOZppAXiLoALA5+x2x6UfggoAawgqAHzO2Uxr928dAIIPQQWAz9GjAsBbBBUAPkePCgBvEVQA+NyftycTVABYQ1AB4HN/Tvjm3zoABB+CCgCf+3MKfZIKAGsIKgB8zu7sUfFzIQCCDkEFgM8ZelQAeImgAsDnnJd+/FsGgCBEUAHgc6yeDMBbBBUAPue8PZmZaQFYRFAB4HNM+AbAWwQVAD7HFPoAvEVQAeBzzEwLwFsEFQA+x8y0ALxFUAHgc3Z6VAB4iaACwPecU+j7twwAwcevQWXkyJGy2WwuH4mJif4sCYAP0KMCwFvh/i6gSZMmWrhwofNxWFiYH6sB4Av0qADwlt+DSnh4OKMoQIgzzEwLwEt+71HZunWrkpKSlJKSor59+2rHjh1n3Tc3N1c5OTkuHwACn53FfgB4ya9BpV27dnrvvfc0b948vf3228rKylKHDh108ODBYvcfM2aM4uLinB/JycllXDEAb7B6MgBv+TWopKWl6frrr1ezZs3UtWtXffHFF5KkqVOnFrv/iBEjdPToUefH7t27y7JcAF6iRwWAt/zeo1JYTEyMmjVrpq1btxb7fFRUlKKiosq4KgAlZc5c8zFc+wFgkd97VArLzc3V5s2bVbNmTX+XAqAUMaICwFt+DSoPPPCAli1bpoyMDH333Xe64YYblJOTo/T0dH+WBaCUsXoyAG/59dLPr7/+qptuuknZ2dmqXr26Lr30Uq1cuVJ169b1Z1kASplhRAWAl/waVKZNm+bPtwdQRljrB4C3AqpHBUBookcFgLcIKgB8rvD8KYyqALCCoALA9wplE0ZVAFhBUAHgc4yoAPAWQQWAz9kZUQHgJYIKAJ8rPKLCej8ArCCoAPA5ogkAbxFUAPicYUQFgJcIKgB8zm4v9Dk5BYAFBBUAPld41WRGVABYQVAB4HOFR1HIKQCsIKgA8DnDPCoAvERQAeBzzKMCwFsEFQA+x10/ALxFUAHgc/SoAPAWQQWAz7HWDwBvEVQA+JyhRwWAlwgqAHyu8Dwqhgn1AVhAUAHgc8xMC8BbBBUAPueyejJJBYAFBBUAPlc4mtBLC8AKggoAn3OZmZYeFQAWEFQA+Bwz0wLwFkEFgM/ZmZkWgJcIKgB8zrjMTEtQAeA5ggoAn3NdPdmPhQAIOgQVAD5HjwoAbxFUAPgcPSoAvEVQAeBzhtWTAXiJoALA5xhRAeAtggoAn2NEBYC3CCoAfK7wbLSMqACwgqACwOcK3+lDTAFgBUEFgM/RowLAWwQVAD7HzLQAvEVQAeBzxmVExY+FAAg6BBUAPmfnrh8AXiKoAPA5elQAeIugAsDnjMtaPwQVAJ4jqADwOZcGWnIKAAsIKgB8jtWTAXiLoALA5+hRAeAtggoAnyscTQgqAKwgqADwucI9KsQUAFZ4HVROnTqlLVu2KD8/vzTrARCC7MxMC8BLloPKyZMndccdd6hChQpq0qSJMjMzJUlDhgzR2LFjS71AAMHPpUfF7sdCAAQdy0FlxIgR+vHHH7V06VKVL1/eub1r166aPn16qRYHIDQwjwoAb4Vb/YLZs2dr+vTpuvTSS2Wz2ZzbGzdurO3bt5dqcQBCg50eFQBesjyicuDAAdWoUcNt+4kTJ1yCCwA4sHoyAG9ZDioXX3yxvvjiC+djRzh5++231b59+9KrDEDIsLN6MgAvWb70M2bMGPXo0UObNm1Sfn6+XnnlFW3cuFHffvutli1b5nUhY8aM0SOPPKKhQ4dq/PjxXr8OgMBjWD0ZgJcsj6h06NBBX3/9tU6ePKn69etr/vz5SkhI0Lfffqs2bdp4VcSqVav01ltvqXnz5l59PYDAxsy0ALxleURFkpo1a6apU6eWSgHHjx9Xv3799Pbbb+uZZ54pldcEEFi46weAtywHFce8KWdTp04dS683aNAgXX311eratet5g0pubq5yc3Odj3Nyciy9FwD/MIXu9SGnALDCclCpV6/eOe/uKSgo8Pi1pk2bpjVr1mjVqlUe7T9mzBiNGjXK49cHEBhcZqblBmUAFlgOKmvXrnV5nJeXp7Vr1+qll17S6NGjPX6d3bt3a+jQoZo/f77LxHHnMmLECA0bNsz5OCcnR8nJyR6/JwD/YGZaAN6yHFRatGjhtq1t27ZKSkrSuHHjdN1113n0Oj/88IP279/v0oBbUFCg5cuXa8KECcrNzVVYWJjL10RFRSkqKspqyQD8jB4VAN7yqpm2OKmpqR5fwpGkK6+8Uhs2bHDZdtttt6lhw4Z66KGH3EIKgOBUdII3cgoAKywHlaINrMYY7d27VyNHjtSFF17o8evExsaqadOmLttiYmJUrVo1t+0AglfRCd7oUQFgheWgUrlyZbdmWmOMkpOTNW3atFIrDEBoKHqph5lpAVhhOagsWbLE5XG5cuVUvXp1NWjQQOHhJbuStHTp0hJ9PYDAU/RSDz0qAKywnCw6duzoizoAhChGVACUhEdB5dNPP/X4Ba+55hqviwEQetwGUBhRAWCBR0Gld+/eHr2YzWazNOEbgNDHiAqAkvAoqNiZoQmAl4rmEnpUAFhhefVkALCiaDAhpwCwwqvbdE6cOKFly5YpMzNTp06dcnluyJAhpVIYgNBgigzIMqICwAqv1vq56qqrdPLkSZ04cUJVq1ZVdna2KlSooBo1ahBUALhgRAVASVi+9HP//ferV69eOnTokKKjo7Vy5Urt2rVLbdq00QsvvOCLGgEEMXpUAJSE5aCybt06DR8+XGFhYQoLC1Nubq6Sk5P1/PPP65FHHvFFjQCCmNuIip/qABCcLAeViIgI5xT6CQkJyszMlCTFxcU5PwcAB/fbk4kqADxnuUelVatWWr16tVJTU9W5c2c98cQTys7O1vvvv69mzZr5okYAwazoooTkFAAWeDyikp+fL0l69tlnVbNmTUnS008/rWrVqmngwIHav3+/3nrrLd9UCSBoFZ3gzc6MbwAs8HhEpWbNmkpPT9ftt9+utm3bSpKqV6+uL7/80mfFAQh+9KgAKAmPR1SGDRumzz77TM2aNVP79u31zjvv6Pjx476sDUAIoEcFQEl4HFRGjBihLVu2aOnSpWrYsKHuu+8+1axZU7fddpu+/vprX9YIIIgVzSVc+QFgheW7fi6//HJNnjxZWVlZGj9+vLZt26bLL79cF110kZ5//nlf1AggiLF6MoCS8Hqtn5iYGN1xxx1asWKFPvvsM2VnZ2vEiBGlWRuAEMDqyQBKwuugcvLkSU2ePFlXXHGFrrnmGlWrVk2jR48uzdoAhAB6VACUhOV5VFasWKHJkydrxowZKigo0A033KBnnnlGV1xxhS/qAxDk3KfQ90sZAIKUx0Hl2Wef1ZQpU7R9+3a1bdtW48aN00033aRKlSr5sj4AQc643Z5MUgHgOY+Dyssvv6x//vOfuuOOO9S0aVNf1gQghBQdQeHKDwArPA4qe/bsUUREhC9rARCC3G5P5toPAAs8bqYlpADwBnf9ACgJr+/6AQBPuE+hT1IB4DmCCgCfKnrphx4VAFYQVAD4lPsU+iQVAJ6zPI+KJNntdm3btk379++X3W53eY75VAAU5nbph5wCwALLQWXlypW6+eabtWvXLrf5EWw2mwoKCkqtOADBj5lpAZSE5aAyYMAAtW3bVl988YVq1qwpm83mi7oAhIiid/lw1w8AKywHla1bt2rGjBlq0KCBL+oBEHKKXvohqQDwnOVm2nbt2mnbtm2+qAVACGJmWgAlYXlE5d5779Xw4cOVlZWlZs2auU0E17x581IrDkDwKzoTLT0qAKywHFSuv/56SdLtt9/u3Gaz2WSMoZkWgBtWTwZQEpaDSkZGhi/qABCi3G9PJqkA8JzloFK3bl1f1AEgRLnNTOufMgAEKa8mfNu+fbvGjx+vzZs3y2azqVGjRho6dKjq169f2vUBCHLMowKgJCzf9TNv3jw1btxY33//vZo3b66mTZvqu+++U5MmTbRgwQJf1AggiLlPoe+fOgAEJ8sjKg8//LDuv/9+jR071m37Qw89pG7dupVacQCCHz0qAErC8ojK5s2bdccdd7htv/3227Vp06ZSKQpA6GD1ZAAlYTmoVK9eXevWrXPbvm7dOtWoUaM0agIQQuhRAVASli/93HXXXbr77ru1Y8cOdejQQTabTV999ZWee+45DR8+3Bc1Aghi7j0qBBUAnrMcVB5//HHFxsbqxRdf1IgRIyRJSUlJGjlypIYMGVLqBQIIbu49Kn4qBEBQshxUbDab7r//ft1///06duyYJCk2NrbUCwMQGlg9GUBJeDWPigMBBcD5cdcPAO95FFRat26tRYsWqUqVKmrVqpVsNttZ912zZk2pFQcg+LmPqBBUAHjOo6By7bXXKioqyvn5uYIKABTm1qPipzoABCePgsqTTz7p/HzkyJG+qgVACKJHBUBJWJ5H5YILLtDBgwfdth85ckQXXHBBqRQFIHQU7UmhRwWAFZaDys6dO1VQUOC2PTc3V7/++mupFAUgdDAzLYCS8Piun08//dT5+bx58xQXF+d8XFBQoEWLFiklJcXSm0+cOFETJ07Uzp07JUlNmjTRE088obS0NEuvAyBwMTMtgJLwOKj07t1b0ul5VNLT012ei4iIUL169fTiiy9aevPatWtr7NixatCggSRp6tSpuvbaa7V27Vo1adLE0msBCEzMTAugJDwOKna7XZKUkpKiVatWKT4+vsRv3qtXL5fHo0eP1sSJE7Vy5UqCChAi3EdU/FQIgKBkecK3jIwMX9ShgoIC/e9//9OJEyfUvn17n7wHgLLnNoBCUAFggeWgMmTIEDVo0MBtXZ8JEyZo27ZtGj9+vKXX27Bhg9q3b68//vhDFStW1KxZs9S4ceNi983NzVVubq7zcU5OjtXyAZQxelQAlITlu34+/vhjXXbZZW7bO3TooBkzZlgu4KKLLtK6deu0cuVKDRw4UOnp6dq0aVOx+44ZM0ZxcXHOj+TkZMvvB6BsFY0lBBUAVlgOKgcPHnS548ehUqVKys7OtlxAZGSkGjRooLZt22rMmDFq0aKFXnnllWL3HTFihI4ePer82L17t+X3A1C2HMEkrJztzGN/VgMg2FgOKg0aNNDcuXPdts+ZM6dUJnwzxrhc3iksKipKlSpVcvkAENgcwSTszNIb5BQAVljuURk2bJgGDx6sAwcOqEuXLpKkRYsW6cUXX7Tcn/LII48oLS1NycnJOnbsmKZNm6alS5cWG4QABCdTeESlgJlpAVhjOajcfvvtys3N1ejRo/X0009LkurVq6eJEyfq1ltvtfRa+/bt0y233KK9e/cqLi5OzZs319y5c9WtWzerZQEIUI5c8uelH4IKAM9ZDiqSNHDgQA0cOFAHDhxQdHS0Klas6NWbv/POO159HYDg4QgmZ3KKzkzJBAAe8SqoOFSvXr206gAQohw9KuFhp1viGE8BYIXlZlrH5ZqkpCSFh4crLCzM5QMACjNF7vqhRwWAFZZHVPr376/MzEw9/vjjqlmzpmxnOvkBoDimyF0/9KgAsMJyUPnqq6+0YsUKtWzZ0gflAAg1RedRIacAsMLypZ/k5GSGbgF4zM5dPwBKwHJQGT9+vB5++GHt3LnTB+UACDXmTPtsOCMqALxg+dJPnz59dPLkSdWvX18VKlRQRESEy/OHDh0qteIABD9HMCnHiAoAL1gOKlZnnwXw12Y/c+2HKfQBeMNyUElPT/dFHQBCFD0qAErCclDJzMw85/N16tTxuhgAocfRo+IMKsxMC8ACy0GlXr1655w7paCgoEQFAQgt9iI9Ktw1CMAKy0Fl7dq1Lo/z8vK0du1avfTSSxo9enSpFQYgNDiCifOuH38WAyDoWA4qLVq0cNvWtm1bJSUlady4cbruuutKpTAAoYHVkwGUhOV5VM4mNTVVq1atKq2XAxAinDPTOqfQ92c1AIKN5RGVnJwcl8fGGO3du1cjR47UhRdeWGqFAQgNRe/6YUAFgBWWg0rlypXdmmmNMUpOTta0adNKrTAAoYHVkwGUhOWgsmTJEpfH5cqVU/Xq1dWgQQOFh1t+OQAhzhFL6FEB4A2Pk8UTTzyhhx9+WB07dpQkHT58WFWqVPFZYQBCg2Nm2nL0qADwgsfNtKNHj9bx48edj+vWrasdO3b4pCgAocMRTMK59APACx4HlaL/uPCPDQBPOO/6CaOZFoB1pXZ7MgCcy5+3J5NUAHjO4x4Vm82mY8eOqXz58jLGyGaz6fjx4263K1eqVKnUiwQQvOxF7vqhRwWAFR4HFWOMUlNTXR63atXK5bHNZmOtHwAuigYVwyT6ACzwOKgUvS0ZADxhijTTMqICwAqPg4rjtmQAsILVkwGUBM20AHzKFFnrh5wCwAqCCgCfcm+mJakA8BxBBYBPmSKLEtKjAsAKggoAnyq6erJEnwoAz3kdVLZt26Z58+bp999/l8Q/PACKV3T15NPb/FUNgGBjOagcPHhQXbt2VWpqqq666irt3btXknTnnXdq+PDhpV4ggODm6EkJLxRU6FMB4CnLQeX+++9XeHi4MjMzVaFCBef2Pn36aO7cuaVaHIDg54gkjtWTJfpUAHjO43lUHObPn6958+apdu3aLtsvvPBC7dq1q9QKAxAaiutRYUQFgKcsj6icOHHCZSTFITs7W1FRUaVSFIDQUfT2ZACwwnJQueKKK/Tee+85H9tsNtntdo0bN06dO3cu1eIAhIAiU+hLjKgA8JzlSz/jxo1Tp06dtHr1ap06dUoPPvigNm7cqEOHDunrr7/2RY0AglhxIyr0qADwlOURlcaNG2v9+vW65JJL1K1bN504cULXXXed1q5dq/r16/uiRgBBzBFUCjfTMp0BAE9ZHlGRpMTERI0aNaq0awEQghyjJ+FhjKgAsM7yiEpKSooef/xxbdmyxRf1AAgxjsETRlQAeMNyULn33ns1d+5cNWrUSG3atNH48eOdk74BQFHFzUzLiAoAT1kOKsOGDdOqVav0888/q2fPnpo4caLq1Kmj7t27u9wNBABSoWZaRlQAeMHrtX5SU1M1atQobdmyRStWrNCBAwd02223lWZtAEKAY/SkXDmbHFmFERUAnvKqmdbh+++/14cffqjp06fr6NGjuuGGG0qrLgAh4s8p9CXbmceMqADwlOWg8ssvv+iDDz7Qhx9+qJ07d6pz584aO3asrrvuOsXGxvqiRgBBzBFKbLbTDbV2YxhRAeAxy0GlYcOGatu2rQYNGqS+ffsqMTHRF3UBCBGF51E5feePkRFJBYBnLAeVn3/+Wampqb6oBUAIclzlsdnoUQFgneVmWkIKACv+HFHRn0GFpALAQx6NqFStWlW//PKL4uPjVaVKFdlsZ18F9dChQ6VWHIDg58gkNtlcJn0DAE94FFRefvllZ6Psyy+/fM6gAgCFmUIjKo6gwurJADzlUVBJT093ft6/f39f1QIgBLn0qJzZxpUfAJ6y3KMSFham/fv3u20/ePCgwsLCLL3WmDFjdPHFFys2NlY1atRQ7969WUMICDH2Qrcn/9lMS1IB4BnLQeVsEzXl5uYqMjLS0mstW7ZMgwYN0sqVK7VgwQLl5+ere/fuOnHihNWyAAQoe6FFCcudWe+HnALAUx7fnvzqq69KOj18++9//1sVK1Z0PldQUKDly5erYcOGlt587ty5Lo8nT56sGjVq6IcfftAVV1xh6bUABKbCM9M6elSYmRaApzwOKi+//LKk0//ATJo0yeUyT2RkpOrVq6dJkyaVqJijR49KOn2XUXFyc3OVm5vrfJyTk1Oi9wPge6bQhG/0qACwyuOgkpGRIUnq3LmzZs6cqSpVqpRqIcYYDRs2TH/729/UtGnTYvcZM2aMRo0aVarvC8C3nP0oNjnvGKRHBYCnLPeoLFmypNRDiiQNHjxY69ev10cffXTWfUaMGKGjR486P3bv3l3qdQAoXXb76T9PT6F/+nNyCgBPWQ4qN9xwg8aOHeu2fdy4cbrxxhu9KuLee+/Vp59+qiVLlqh27dpn3S8qKkqVKlVy+QAQ2FxWT+auHwAWWQ4qy5Yt09VXX+22vUePHlq+fLml1zLGaPDgwZo5c6YWL16slJQUq+UACHDGbVFCRlQAeM7yooTHjx8v9jbkiIgIy82tgwYN0ocffqhPPvlEsbGxysrKkiTFxcUpOjraamkAApBzHhUVuuuH1ZMBeMjyiErTpk01ffp0t+3Tpk1T48aNLb3WxIkTdfToUXXq1Ek1a9Z0fhT3+gCCk53VkwGUgOURlccff1zXX3+9tm/fri5dukiSFi1apI8++kj/+9//LL0WcykAoc8Ut3oyP/sAPGQ5qFxzzTWaPXu2nn32Wc2YMUPR0dFq3ry5Fi5cqI4dO/qiRgBBrPBaP0z4BsAqy0FFkq6++upiG2oBoCh7Masnk1MAeMpyj4okHTlyRP/+97/1yCOP6NChQ5KkNWvW6LfffivV4gAEP0cmoUcFgDcsj6isX79eXbt2VVxcnHbu3Kk777xTVatW1axZs7Rr1y699957vqgTQJAqPKJiK7INAM7H8ojKsGHD1L9/f23dulXly5d3bk9LS7M8jwqA0OeYmbZwjwpBBYCnLAeVVatW6V//+pfb9lq1ajnnQQEAB1NMjwrTqADwlOWgUr58+WIndtuyZYuqV69eKkUBCB1/TqFPjwoA6ywHlWuvvVZPPfWU8vLyJJ0ezs3MzNTDDz+s66+/vtQLBBDcnDPTsnoyAC9YDiovvPCCDhw4oBo1auj3339Xx44d1aBBA8XGxmr06NG+qBFAEHPOTKs/V08mqADwlOW7fipVqqSvvvpKixcv1po1a2S329W6dWt17drVF/UBCHLOHpVyhdf6AQDPeDXhmyR16dLFOYU+AJyNY/CkcI8KM9MC8JRHQeXVV1/V3XffrfLly+vVV189574VK1ZUkyZN1K5du1IpEEBwK7x6srNHxe7HggAEFY+Cyssvv6x+/fqpfPnyevnll8+5b25urvbv36/7779f48aNK5UiAQQvu8taP6c/ZzwFgKc8CioZGRnFfn42CxYs0M0330xQAVDsPCo00wLwlFdr/ZzP3/72Nz322GO+eGkAQcalR8W5jaACwDNeBZVFixapZ8+eql+/vho0aKCePXtq4cKFzuejo6M1dOjQUisSQPAqPI/KnyMq/qwIQDCxHFQmTJigHj16KDY2VkOHDtWQIUNUqVIlXXXVVZowYYIvagQQxOzF3vXjv3oABBfLtyePGTNGL7/8sgYPHuzcNmTIEF122WUaPXq0y3YAMCo8M+3pbfSoAPCU5RGVnJwc9ejRw2179+7di10DCMBfW+ERFZppAVhlOahcc801mjVrltv2Tz75RL169SqVogCEDlNMjwo5BYCnPJ7wzaFRo0YaPXq0li5dqvbt20uSVq5cqa+//lrDhw/3TZUAglaxPSrMpALAQx5P+FZYlSpVtGnTJm3atMm5rXLlynr33Xe5LRmAC1Pc6snMTAvAQ5YnfAMAK1x7VBzbGFEB4BmvJ3zLzs7WwYMHS7MWACGm8MRuNrF6MgDrLAWVI0eOaNCgQYqPj1dCQoJq1Kih+Ph4DR48WEeOHPFRiQCCVeGBE2amBeANj+dROXTokNq3b6/ffvtN/fr1U6NGjWSM0ebNmzVlyhQtWrRI33zzjapUqeLLegEEkcKXeE430zIzLQBrPA4qTz31lCIjI7V9+3YlJCS4Pde9e3c99dRT511dGcBfh0sgsYkeFQCWeXzpZ/bs2XrhhRfcQookJSYm6vnnny92fhUAf12uIyrMowLAOo+Dyt69e9WkSZOzPt+0aVNlZWWVSlEAQo/rWj8kFQCe8TioxMfHa+fOnWd9PiMjQ9WqVSuNmgCEiKI9KqyeDMAqj4NKjx499Oijj+rUqVNuz+Xm5urxxx8vdg0gAH9dhQMJixIC8IbHzbSjRo1S27ZtdeGFF2rQoEFq2LChJGnTpk164403lJubq/fff99nhQIIPoUDCWv9APCGx0Gldu3a+vbbb3XPPfdoxIgRhabFtqlbt26aMGGCkpOTfVYogODjNo8KIyoALPI4qEhSSkqK5syZo8OHD2vr1q2SpAYNGqhq1ao+KQ5AcDNn6VEhpwDwlKWg4lClShVdcsklpV0LgBDj0qMisXoyAMu8XusHAM7HFOlRcUyiz10/ADxFUAHgM45AcvqOH1ZPBmAdQQWAzzib7s88pkcFgFUEFQA+4xhRcQSUcmf+xWFmWgCeIqgA8BlH06wjqIgeFQAWEVQA+EzhHhWJ1ZMBWEdQAeAzdrtjYsjTj1nrB4BVBBUAPufsUXFcAWJEBYCHCCoAfMZxiccRVGyMqACwiKACwGeK9qiw1g8AqwgqAHzGfrZ5VPxUD4DgQ1AB4DOOgZNyZ5pTHIGFERUAniKoAPAZU6RHxRFYyCkAPEVQAeAzzh6VM4+dPSp00wLwEEEFgM84e1SctyfTowLAGr8GleXLl6tXr15KSkqSzWbT7Nmz/VkOgFLm7FFx3PVzZjs9KgA85degcuLECbVo0UITJkzwZxkAfKToPCqsngzAqnB/vnlaWprS0tL8WQIAHzKs9QOghPwaVKzKzc1Vbm6u83FOTo4fqwFwPm6rJzOiAsCioGqmHTNmjOLi4pwfycnJ/i4JwDmwejKAkgqqoDJixAgdPXrU+bF7925/lwTgHM7Wo8LdyQA8FVSXfqKiohQVFeXvMgB4yDhvTz792Ll6MjcoA/BQUI2oAAguf96eXGT1ZLu/KgIQbPw6onL8+HFt27bN+TgjI0Pr1q1T1apVVadOHT9WBqA0sHoygJLya1BZvXq1Onfu7Hw8bNgwSVJ6erqmTJnip6oAlBZ6VACUlF+DSqdOnZzXsAGEHucU+mceO3pUDD0qADxEjwoA3ynaoyLmUQFgDUEFgM/QowKgpAgqAHzmbKsn06MCwFMEFQA+48gj5YqMqNCbBsBTBBUAPsPqyQBKiqACwGeMM6jI5U96VAB4iqACwGecM9AWmZmWnALAUwQVAD5zth4VRlQAeIqgAsBnmJkWQEkRVAD4zNl6VLjrB4CnCCoIedv2H9f0VZmy89/4Muec8E1FelT8VRCAoOPXtX6AsvDEJz/pm+0HlVy1gjrUj/d3OX8ppujMtGe206MCwFOMqCDk7cv5Q5J04Fiunyv566FHBUBJEVQQ8k7kFkiSjufm+7mSv54/p9A//bjcmX9x6FEB4CmCCkKeI6Ac/4Og4i+sngzAWwQVhDS73fwZVBhRKXNFR1SYRwWAVQQVhLSTeQXOzwkqZc8xM617jwpBBYBnCCoIaYUv93Dpp+y59ajQTAvAIoIKQtrx3LxCnxNUytqfU+g75lEp8gQAnAdBBSHtWOERFYJKmWP1ZAAlRVBBSHPcmiwRVPzBOTNtkdWTCSoAPEVQQUhzufRDj0qZc/aonHnMlR8AVhFUENK49ONfjoETZqYF4C2CCkLaiVyCij85e1TO/EvDzLQArCKoIKQdLxJU+AVZts62ejI9KgA8RVBBSDtWKKgYI508VXCOvVHaTNGZaZ3b/VMPgOBDUEFIO1Hkck/Rx/AtOz0qAEqIoIKQVvROn2MElTJld5tHxbEoIUkFgGcIKghpRRtouUW5bJki86gw4RsAqwgqCGnHigQT7vwpW0auPSqOJhVyCgBPEVQQ0k6cIqj409l7VEgqADxDUEFIc1zqqRAZ5vIYZePsPSr+qghAsCGoIKQ5RlAS48q7PEbZMG7zqJzZ7qd6AAQfggpCmjOoVCKo+IPbzLQ00wKwiKCCkJVXYNcfeXZJjKj4C6snAygpggpCVuHJ3RIcIyr0qJSps/Wo2O3+qghAsCGoIGQ5bk2OCi+nKhUiJDGiUtbcelT8WAuA4ERQQchy3JocWz5cFaMIKv5gzjaiwqUfAB4iqCBkOS7zVIwKV8Xy4S7bUDbce1Qc2wkqADxDUEHIcqzrExMVropRZ+ZRYUSlTNmLrJ7MooQArCKoIGQ5mmkrRv156YfVk8uWI4+UKzKiwoAKAE8RVBCyHJd5TveonL70w+rJZYvVkwGUFEEFIeu4y6UfelT8wbit9XP6MT0qADxFUEHIOp7r3kz7e16BCmiQKDPmz/uTT//BFPoALCKoIGQ57/opH66YM820Eg21Zano6snOmWkJiwA8RFBByHKOqESGKyo8TJFh5Vy2w/dYPRlASRFUELIcjbOOyz7MpVL26FEBUFIEFYSswrcnF/6TEZWy4+hRcUyd75hKn5gCwFMEFYSswrcnSwQVf2BmWgAlRVBByCp8e7IkblH2gz97VM5c+inHzLQArPF7UHnjjTeUkpKi8uXLq02bNlqxYoW/S0KIOF700s+ZkRVmpy07f46onPnzzHYmfIOvbD9wnJ/xEOPXoDJ9+nTdd999evTRR7V27VpdfvnlSktLU2Zmpj/LQohwBJWil36YnbYscdcPys7rS7bpyheX6e/jl2vXwRP+LgelJNyfb/7SSy/pjjvu0J133ilJGj9+vObNm6eJEydqzJgxfqvreG6+so/l+u39UTocl3gcl34cf/56+KR2ZvOPWFnI+f30ObAVueunwBjOAUrVR6sy9eayHZKkXw//rhsmfavXb26tGrFRfq4s+FWIClON2PJ+e3+/BZVTp07phx9+0MMPP+yyvXv37vrmm2+K/Zrc3Fzl5v4ZIHJycnxS2+Kf92vIR2t98tooe46A4hhZmfz1Tk3+eqcfK/rrcV76KTSi0umFpf4rCCFryJUXav7GLP2cdUz/ePNbf5cTEq5pkaRXb2rlt/f3W1DJzs5WQUGBEhISXLYnJCQoKyur2K8ZM2aMRo0a5fPaIsrZFBvl18EmlJIrUqs7z2W3xgn67Mc9NNOWsUrREbriwuqSpPiKkep8UXWt3nnYz1Uh1FSICtOwbqnqc3Ed3X5ZPQ37749alXHI32WFhPIR/m1ntRk/dbXt2bNHtWrV0jfffKP27ds7t48ePVrvv/++fv75Z7evKW5EJTk5WUePHlWlSpXKpG4AAFAyOTk5iouL8+j3t9+GDeLj4xUWFuY2erJ//363URaHqKgoRUVxvREAgL8Kv43nREZGqk2bNlqwYIHL9gULFqhDhw5+qgoAAAQSvzZiDBs2TLfccovatm2r9u3b66233lJmZqYGDBjgz7IAAECA8GtQ6dOnjw4ePKinnnpKe/fuVdOmTfXll1+qbt26/iwLAAAECL8105YGK804AAAgMFj5/e33KfQBAADOhqACAAACFkEFAAAELIIKAAAIWAQVAAAQsAgqAAAgYBFUAABAwCKoAACAgEVQAQAAAcuvU+iXlGNS3ZycHD9XAgAAPOX4ve3J5PhBHVSOHTsmSUpOTvZzJQAAwKpjx44pLi7unPsE9Vo/drtde/bsUWxsrGw2W4lfLycnR8nJydq9e3fIrh3EMQa/UD8+iWMMBaF+fBLHWBLGGB07dkxJSUkqV+7cXShBPaJSrlw51a5du9Rft1KlSiH7TefAMQa/UD8+iWMMBaF+fBLH6K3zjaQ40EwLAAACFkEFAAAELIJKIVFRUXryyScVFRXl71J8hmMMfqF+fBLHGApC/fgkjrGsBHUzLQAACG2MqAAAgIBFUAEAAAGLoAIAAAIWQQUAAAQsgkohb7zxhlJSUlS+fHm1adNGK1as8HdJXhkzZowuvvhixcbGqkaNGurdu7e2bNnisk///v1ls9lcPi699FI/VWzdyJEj3epPTEx0Pm+M0ciRI5WUlKTo6Gh16tRJGzdu9GPF1tWrV8/tGG02mwYNGiQp+M7h8uXL1atXLyUlJclms2n27Nkuz3tyznJzc3XvvfcqPj5eMTExuuaaa/Trr7+W4VGc27mOMS8vTw899JCaNWummJgYJSUl6dZbb9WePXtcXqNTp05u57Vv375lfCRnd77z6Mn3ZSCfx/MdX3E/kzabTePGjXPuE8jn0JPfD4H2s0hQOWP69Om677779Oijj2rt2rW6/PLLlZaWpszMTH+XZtmyZcs0aNAgrVy5UgsWLFB+fr66d++uEydOuOzXo0cP7d271/nx5Zdf+qli7zRp0sSl/g0bNjife/755/XSSy9pwoQJWrVqlRITE9WtWzfn+lDBYNWqVS7Ht2DBAknSjTfe6NwnmM7hiRMn1KJFC02YMKHY5z05Z/fdd59mzZqladOm6auvvtLx48fVs2dPFRQUlNVhnNO5jvHkyZNas2aNHn/8ca1Zs0YzZ87UL7/8omuuucZt37vuusvlvL755ptlUb5HzncepfN/XwbyeTzf8RU+rr179+rdd9+VzWbT9ddf77JfoJ5DT34/BNzPooExxphLLrnEDBgwwGVbw4YNzcMPP+ynikrP/v37jSSzbNky57b09HRz7bXX+q+oEnryySdNixYtin3ObrebxMREM3bsWOe2P/74w8TFxZlJkyaVUYWlb+jQoaZ+/frGbrcbY4L7HEoys2bNcj725JwdOXLEREREmGnTpjn3+e2330y5cuXM3Llzy6x2TxU9xuJ8//33RpLZtWuXc1vHjh3N0KFDfVtcKSnuGM/3fRlM59GTc3jttdeaLl26uGwLpnNY9PdDIP4sMqIi6dSpU/rhhx/UvXt3l+3du3fXN99846eqSs/Ro0clSVWrVnXZvnTpUtWoUUOpqam66667tH//fn+U57WtW7cqKSlJKSkp6tu3r3bs2CFJysjIUFZWlsv5jIqKUseOHYP2fJ46dUr/+c9/dPvtt7sswBns59DBk3P2ww8/KC8vz2WfpKQkNW3aNGjP69GjR2Wz2VS5cmWX7R988IHi4+PVpEkTPfDAA0E1Eiid+/sylM7jvn379MUXX+iOO+5wey5YzmHR3w+B+LMY1IsSlpbs7GwVFBQoISHBZXtCQoKysrL8VFXpMMZo2LBh+tvf/qamTZs6t6elpenGG29U3bp1lZGRoccff1xdunTRDz/8EBSzLLZr107vvfeeUlNTtW/fPj3zzDPq0KGDNm7c6DxnxZ3PXbt2+aPcEps9e7aOHDmi/v37O7cF+zkszJNzlpWVpcjISFWpUsVtn2D8Of3jjz/08MMP6+abb3ZZ7K1fv35KSUlRYmKifvrpJ40YMUI//vij89JfoDvf92UoncepU6cqNjZW1113ncv2YDmHxf1+CMSfRYJKIYX/pyqdPolFtwWbwYMHa/369frqq69ctvfp08f5edOmTdW2bVvVrVtXX3zxhdsPXSBKS0tzft6sWTO1b99e9evX19SpU52Ne6F0Pt955x2lpaUpKSnJuS3Yz2FxvDlnwXhe8/Ly1LdvX9ntdr3xxhsuz911113Oz5s2baoLL7xQbdu21Zo1a9S6deuyLtUyb78vg/E8vvvuu+rXr5/Kly/vsj1YzuHZfj9IgfWzyKUfSfHx8QoLC3NLgvv373dLlcHk3nvv1aeffqolS5aodu3a59y3Zs2aqlu3rrZu3VpG1ZWumJgYNWvWTFu3bnXe/RMq53PXrl1auHCh7rzzznPuF8zn0JNzlpiYqFOnTunw4cNn3ScY5OXl6R//+IcyMjK0YMECl9GU4rRu3VoRERFBeV4l9+/LUDmPK1as0JYtW877cykF5jk82++HQPxZJKhIioyMVJs2bdyG5RYsWKAOHTr4qSrvGWM0ePBgzZw5U4sXL1ZKSsp5v+bgwYPavXu3atasWQYVlr7c3Fxt3rxZNWvWdA65Fj6fp06d0rJly4LyfE6ePFk1atTQ1Vdffc79gvkcenLO2rRpo4iICJd99u7dq59++ilozqsjpGzdulULFy5UtWrVzvs1GzduVF5eXlCeV8n9+zIUzqN0epSzTZs2atGixXn3DaRzeL7fDwH5s1jq7blBatq0aSYiIsK88847ZtOmTea+++4zMTExZufOnf4uzbKBAweauLg4s3TpUrN3717nx8mTJ40xxhw7dswMHz7cfPPNNyYjI8MsWbLEtG/f3tSqVcvk5OT4uXrPDB8+3CxdutTs2LHDrFy50vTs2dPExsY6z9fYsWNNXFycmTlzptmwYYO56aabTM2aNYPm+BwKCgpMnTp1zEMPPeSyPRjP4bFjx8zatWvN2rVrjSTz0ksvmbVr1zrvePHknA0YMMDUrl3bLFy40KxZs8Z06dLFtGjRwuTn5/vrsFyc6xjz8vLMNddcY2rXrm3WrVvn8rOZm5trjDFm27ZtZtSoUWbVqlUmIyPDfPHFF6Zhw4amVatWQXGMnn5fBvJ5PN/3qTHGHD161FSoUMFMnDjR7esD/Rye7/eDMYH3s0hQKeT11183devWNZGRkaZ169Yut/MGE0nFfkyePNkYY8zJkydN9+7dTfXq1U1ERISpU6eOSU9PN5mZmf4t3II+ffqYmjVrmoiICJOUlGSuu+46s3HjRufzdrvdPPnkkyYxMdFERUWZK664wmzYsMGPFXtn3rx5RpLZsmWLy/ZgPIdLliwp9vsyPT3dGOPZOfv999/N4MGDTdWqVU10dLTp2bNnQB3zuY4xIyPjrD+bS5YsMcYYk5mZaa644gpTtWpVExkZaerXr2+GDBliDh486N8DK+Rcx+jp92Ugn8fzfZ8aY8ybb75poqOjzZEjR9y+PtDP4fl+PxgTeD+LtjOFAwAABBx6VAAAQMAiqAAAgIBFUAEAAAGLoAIAAAIWQQUAAAQsggoAAAhYBBUAABCwCCoAgkq9evU0fvx4f5cBoIwQVACcVf/+/dW7d29JUqdOnXTfffeV2XtPmTJFlStXdtu+atUq3X333WVWBwD/Cvd3AQD+Wk6dOqXIyEivv7569eqlWA2AQMeICoDz6t+/v5YtW6ZXXnlFNptNNptNO3fulCRt2rRJV111lSpWrKiEhATdcsstys7Odn5tp06dNHjwYA0bNkzx8fHq1q2bJOmll15Ss2bNFBMTo+TkZN1zzz06fvy4JGnp0qW67bbbdPToUef7jRw5UpL7pZ/MzExde+21qlixoipVqqR//OMf2rdvn/P5kSNHqmXLlnr//fdVr149xcXFqW/fvjp27JhznxkzZqhZs2aKjo5WtWrV1LVrV504ccJHf5sArCCoADivV155Re3bt9ddd92lvXv3au/evUpOTtbevXvVsWNHtWzZUqtXr9bcuXO1b98+/eMf/3D5+qlTpyo8PFxff/213nzzTUlSuXLl9Oqrr+qnn37S1KlTtXjxYj344IOSpA4dOmj8+PGqVKmS8/0eeOABt7qMMerdu7cOHTqkZcuWacGCBdq+fbv69Onjst/27ds1e/Zsff755/r888+1bNkyjR07VtLp5elvuukm3X777dq8ebOWLl2q6667TiyDBgQGLv0AOK+4uDhFRkaqQoUKSkxMdG6fOHGiWrdurWeffda57d1331VycrJ++eUXpaamSpIaNGig559/3uU1C/e7pKSk6Omnn9bAgQP1xhtvKDIyUnFxcbLZbC7vV9TChQu1fv16ZWRkKDk5WZL0/vvvq0mTJlq1apUuvvhiSZLdbteUKVMUGxsrSbrlllu0aNEijR49Wnv37lV+fr6uu+461a1bV5LUrFmzEvxtAShNjKgA8NoPP/ygJUuWqGLFis6Phg0bSjo9iuHQtm1bt69dsmSJunXrplq1aik2Nla33nqrDh48aOmSy+bNm5WcnOwMKZLUuHFjVa5cWZs3b3Zuq1evnjOkSFLNmjW1f/9+SVKLFi105ZVXqlmzZrrxxhv19ttv6/Dhw57/JQDwKYIKAK/Z7Xb16tVL69atc/nYunWrrrjiCud+MTExLl+3a9cuXXXVVWratKk+/vhj/fDDD3r99dclSXl5eR6/vzFGNpvtvNsjIiJcnrfZbLLb7ZKksLAwLViwQHPmzFHjxo312muv6aKLLlJGRobHdQDwHYIKAI9ERkaqoKDAZVvr1q21ceNG1atXTw0aNHD5KBpOClu9erXy8/P14osv6tJLL1Vqaqr27Nlz3vcrqnHjxsrMzNTu3bud2zZt2qSjR4+qUaNGHh+bzWbTZZddplGjRmnt2rWKjIzUrFmzPP56AL5DUAHgkXr16um7777Tzp07lZ2dLbvdrkGDBunQoUO66aab9P3332vHjh2aP3++br/99nOGjPr16ys/P1+vvfaaduzYoffff1+TJk1ye7/jx49r0aJFys7O1smTJ91ep2vXrmrevLn69eunNWvW6Pvvv9ett96qjh07Fnu5qTjfffednn32Wa1evVqZmZmaOXOmDhw4YCnoAPAdggoAjzzwwAMKCwtT48aNVb16dWVmZiopKUlff/21CgoK9Pe//11NmzbV0KFDFRcXp3Llzv7PS8uWLfXSSy/pueeeU9OmTfXBBx9ozJgxLvt06NBBAwYMUJ8+fVS9enW3Zlzp9EjI7NmzVaVKFV1xxRXq2rWrLrjgAk2fPt3j46pUqZKWL1+uq666SqmpqXrsscf04osvKi0tzfO/HAA+YzPcgwcAAAIUIyoAACBgEVQAAEDAIqgAAICARVABAAABi6ACAAACFkEFAAAELIIKAAAIWAQVAAAQsAgqAAAgYBFUAABAwCKoAACAgEVQAQAAAev/AyAGSqmuBNG5AAAAAElFTkSuQmCC"></div></div><div class="jp-OutputArea-promptOverlay" title="Toggle output scrolling"></div></div></div><div class="lm-Widget jp-CellFooter jp-Cell-footer"></div></div><div class="lm-Widget jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs" aria-label="Code Cell Content" data-windowed-list-index="6"><div class="lm-Widget jp-CellHeader jp-Cell-header"></div><div class="lm-Widget lm-Panel jp-Cell-inputWrapper"><div class="lm-Widget jp-Collapser jp-InputCollapser jp-Cell-inputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-InputArea jp-Cell-inputArea"><div class="lm-Widget jp-InputPrompt jp-InputArea-prompt">[ ]:</div><div class="lm-Widget jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline"><div class="cm-editor ͼ1 ͼ2 ͼ4 ͼo" style=""><div aria-live="polite" style="position: fixed; top: -10000px;"></div><div tabindex="-1" class="cm-scroller"><div spellcheck="false" autocorrect="off" autocapitalize="off" translate="no" contenteditable="true" class="cm-content" style="tab-size: 4;" role="textbox" aria-multiline="true" data-language="python"><div class="cm-line"><br></div><div class="cm-line">GOR <span class="ͼ10">=</span> <span class="ͼ19">(</span><span class="ͼu">2.293</span> × <span class="ͼu">10</span><span class="ͼ10">^</span><span class="ͼ19">(</span><span class="ͼ10">-</span><span class="ͼu">5</span><span class="ͼ19">)</span> × <span class="ͼ19">(</span>p × 10<span class="ͼ10">^</span><span class="ͼ19">(</span><span class="ͼ10">-</span><span class="ͼu">0.00091</span> × API<span class="ͼ19">)</span><span class="ͼ19">)</span> × <span class="ͼ19">(</span>T<span class="ͼ10">^</span><span class="ͼ19">(</span><span class="ͼ10">-</span><span class="ͼu">1.187</span><span class="ͼ19">)</span><span class="ͼ19">)</span> × <span class="ͼ19">(</span>p<span class="ͼ10">^</span><span class="ͼu">0.5</span><span class="ͼ19">)</span><span class="ͼ19">)</span> <span class="ͼ10">+</span> <span class="ͼu">0.00091</span></div><div class="cm-line"><br></div><div class="cm-line"><br></div><div class="cm-line"><br></div><div class="cm-line"><br></div></div><div class="cm-layer cm-layer-above cm-cursorLayer" aria-hidden="true" style="z-index: 150; animation-duration: 1200ms; animation-name: cm-blink;"><div class="cm-cursor cm-cursor-primary" style="left: 6px; top: 41.7292px; height: 15.3333px;"></div></div><div class="cm-layer cm-selectionLayer" aria-hidden="true" style="z-index: -2;"></div></div></div></div></div></div><div class="lm-Widget jp-CellResizeHandle"></div><div class="lm-Widget lm-Panel jp-Cell-outputWrapper"><div class="lm-Widget jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser"><div class="jp-Collapser-child"></div></div><div class="lm-Widget jp-OutputArea jp-Cell-outputArea"><div class="jp-OutputArea-promptOverlay" title="Toggle output scrolling"></div></div></div><div class="lm-Widget jp-CellFooter jp-Cell-footer"></div></div></div></div><button class="lm-Widget jp-Notebook-footer" data-jp-traversable="true">Click to add a cell.</button></div></div></div><div class="lm-Widget" id="spacer-widget-bottom" style="position: absolute; contain: strict; top: 638px; left: 0px; width: 1280px; height: 0px;"></div></div><div class="lm-SplitPanel-handle lm-mod-hidden" style="position: absolute; contain: style; top: 0px; left: 1280px; width: 1px; height: 638px;"></div><div class="lm-Widget lm-Panel lm-mod-hidden lm-SplitPanel-child" id="jp-right-stack" role="complementary" style="position: absolute; contain: strict;"><button class="jp-Button jp-SidePanel-collapse lm-Widget" title="Collapse right side panel"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" data-icon="ui-components:close" data-icon-id="0f44268c-bc5d-4ba4-8d4f-e9f87f400810"><g class="jp-icon-none jp-icon-selectable-inverse jp-icon3-hover" fill="none"><circle cx="12" cy="12" r="11"></circle></g><g class="jp-icon3 jp-icon-selectable jp-icon-accent2-hover" fill="#616161"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"></path></g><g class="jp-icon-none jp-icon-busy" fill="none"><circle cx="12" cy="12" r="7"></circle></g></svg></button><div class="lm-Widget lm-Panel lm-StackedPanel" style="min-width: 0px; min-height: 0px;"><div class="lm-Widget jp-NotebookTools lm-mod-hidden lm-StackedPanel-child" id="notebook-tools" style="position: absolute; contain: strict;"><div class="lm-Widget jp-Collapse"><div class="lm-Widget jp-Collapse-header jp-Collapse-header-collapsed"><div class="jp-Collapser-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 18 18" data-icon="ui-components:caret-down" data-icon-id="f487de1d-5477-4b68-a71b-86af0e6134cd"><g class="jp-icon3" fill="#616161" shape-rendering="geometricPrecision"><path d="M5.2,7.5L9,11.2l3.8-3.8H5.2z"></path></g></svg></div><span class="jp-Collapser-title">Common Tools</span></div><div class="lm-Widget lm-Panel jp-Collapse-contents lm-mod-hidden"><div class="lm-Widget jp-RankedPanel"><div class="lm-Widget jp-NotebookTools-tool"><div class="lm-Widget jp-MetadataForm-placeholder"><div class="jp-MetadataForm-placeholderContent">No metadata.</div></div></div></div></div></div><div class="lm-Widget jp-Collapse"><div class="lm-Widget jp-Collapse-header jp-Collapse-header-collapsed"><div class="jp-Collapser-icon"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 18 18" data-icon="ui-components:caret-down" data-icon-id="f487de1d-5477-4b68-a71b-86af0e6134cd"><g class="jp-icon3" fill="#616161" shape-rendering="geometricPrecision"><path d="M5.2,7.5L9,11.2l3.8-3.8H5.2z"></path></g></svg></div><span class="jp-Collapser-title">Advanced Tools</span></div><div class="lm-Widget lm-Panel jp-Collapse-contents lm-mod-hidden"><div class="lm-Widget jp-RankedPanel"><div class="lm-Widget jp-NotebookTools-tool"><div class="lm-Widget jp-MetadataForm-placeholder"><div class="jp-MetadataForm-placeholderContent">No metadata.</div></div></div></div></div></div></div><div class="lm-Widget jp-SidePanel as-widget-container lm-mod-hidden lm-StackedPanel-child" id="anaconda-assistant" style="position: absolute; contain: strict;"><div class="lm-Widget lm-Panel as-widget-content" role="region" aria-label="Assistant content"><div class="lm-Widget as-widget-content"><div class="as-container"><div class="as-login-container"><div class="as-login-header"><div class="as-login-icon"><svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg"><g><path d="M3.69338 15.348V15.2893C3.69338 14.6966 3.75197 14.1333 3.84112 13.5713V13.5126L3.78253 13.4832C3.25018 13.2763 2.7484 12.9799 2.24662 12.713L2.18803 12.6836L2.15874 12.7423C1.83398 13.3938 1.56781 14.1052 1.33093 14.8167L1.30164 14.8754L1.36022 14.9048C2.09888 15.1117 2.83755 15.2599 3.60551 15.3199L3.69466 15.3493L3.69338 15.348Z" fill="#3DAE2B"></path><path d="M5.22926 5.18455C5.22926 5.16496 5.22926 5.16496 5.22926 5.18455C4.78606 5.18455 4.37215 5.21393 3.92896 5.24331C3.98754 5.68781 4.0474 6.10293 4.16584 6.54743C4.4613 6.04417 4.81535 5.5703 5.22926 5.18455Z" fill="#3DAE2B"></path><path d="M3.69337 15.8807V15.8219H3.63479C3.01456 15.7632 2.36505 15.6444 1.71426 15.4962L1.53723 15.4669L1.62638 15.615C2.18802 16.4746 2.89612 17.2449 3.72394 17.9257L3.84238 18.0445V17.8375C3.75323 17.1555 3.69464 16.4746 3.69464 15.882L3.69337 15.8807Z" fill="#3DAE2B"></path><path d="M6.79449 0.533936C6.02653 0.800892 5.28787 1.15598 4.60779 1.60048C5.10957 1.68989 5.61263 1.80741 6.11441 1.95557C6.32073 1.48169 6.52832 1.00781 6.79449 0.533936Z" fill="#3DAE2B"></path><path d="M10.0153 0C9.6307 0 9.24736 0.029378 8.89331 0.0587559C9.42566 0.413846 9.92744 0.828969 10.3999 1.27347L10.7845 1.62856L10.3999 2.01431C10.0752 2.31064 9.75042 2.66573 9.45495 3.0221V3.05148C9.45495 3.05148 9.39637 3.11023 9.30722 3.22902C9.5441 3.19964 9.77971 3.19964 10.0166 3.19964C13.7685 3.19964 16.811 6.25112 16.811 10.0141C16.811 13.777 13.7685 16.8285 10.0166 16.8285C8.71628 16.8285 7.50513 16.4734 6.471 15.8207C5.96921 15.8794 5.43686 15.9101 4.93508 15.9101C4.6982 15.9101 4.46259 15.9101 4.22571 15.8807C4.255 16.6509 4.31486 17.4505 4.43202 18.2808C6.0278 19.3767 7.94705 19.9987 10.0153 19.9987C15.54 19.9987 20 15.5243 20 9.98467C20 4.44501 15.5387 0 10.0153 0Z" fill="#3DAE2B"></path><path d="M8.68565 2.37066C8.89197 2.13308 9.09956 1.89678 9.30588 1.68858C8.83339 1.30283 8.3316 0.918363 7.82855 0.592651C7.50379 1.12656 7.23761 1.68858 7.00073 2.25187C7.44393 2.42941 7.88713 2.60696 8.33033 2.81516C8.50736 2.57758 8.65509 2.40003 8.68438 2.37066H8.68565Z" fill="#3DAE2B"></path><path d="M1.92056 8.08915L1.94985 8.1479L2.00843 8.11852C2.48093 7.79281 2.95342 7.4671 3.45647 7.20014L3.51506 7.17077V7.11201C3.36732 6.54872 3.24888 5.95605 3.1903 5.36338V5.30463H3.13171C2.39305 5.45279 1.65438 5.63034 0.975576 5.89729L0.916992 5.92667L0.946284 5.98543C1.1526 6.69689 1.50792 7.40707 1.92056 8.08915Z" fill="#3DAE2B"></path><path d="M1.80213 9.03692L1.74355 9.09568C1.2112 9.54018 0.70942 10.0141 0.236929 10.548L0.178345 10.6067L0.236929 10.6655C0.769277 11.11 1.30035 11.5251 1.89128 11.9096L1.94987 11.939L1.97916 11.8802C2.24533 11.4063 2.5408 10.9618 2.86556 10.5173L2.89485 10.4586L2.86556 10.4292C2.51151 10.0141 2.15619 9.59894 1.83143 9.12506L1.80213 9.03564V9.03692Z" fill="#3DAE2B"></path><path d="M5.73103 15.3774H5.90806L5.76032 15.2586C5.22797 14.8141 4.75548 14.3109 4.40143 13.7182V13.6888L4.28299 13.6301V13.7195C4.22441 14.2227 4.16455 14.7566 4.16455 15.2893V15.348H4.22313C4.46002 15.348 4.69563 15.3774 4.93251 15.3774H5.72976H5.73103Z" fill="#3DAE2B"></path><path d="M5.40623 4.29686C5.52467 3.7936 5.64311 3.31972 5.82014 2.84457C5.19991 2.66702 4.5504 2.51885 3.89961 2.42944C3.84102 3.11152 3.84102 3.73357 3.87031 4.38499C4.3721 4.32624 4.87515 4.29558 5.40623 4.29558V4.29686Z" fill="#3DAE2B"></path><path d="M6.3512 4.26617C6.7944 3.99921 7.26689 3.76291 7.73938 3.58409C7.38533 3.40654 7.03 3.25838 6.67595 3.13959C6.55751 3.49468 6.43907 3.88042 6.3512 4.26489V4.26617Z" fill="#3DAE2B"></path><path d="M1.71294 12.3847H1.65436C1.15258 12.059 0.649521 11.7027 0.17703 11.3182L0.0292969 11.1994L0.0585888 11.3769C0.17703 12.3247 0.443204 13.2737 0.855838 14.1615L0.914422 14.3096L0.973006 14.1615C1.17932 13.5982 1.4162 13.0362 1.68238 12.5023L1.71167 12.3835L1.71294 12.3847Z" fill="#3DAE2B"></path><path d="M2.98391 2.90332C2.39298 3.49598 1.86191 4.17679 1.448 4.88824C1.94978 4.74008 2.48213 4.62129 3.0132 4.5025C2.98391 3.96859 2.98391 3.43595 2.98391 2.90204V2.90332Z" fill="#3DAE2B"></path><path d="M3.27938 9.92594V9.7484C3.30867 9.12635 3.39782 8.53368 3.57484 7.97039L3.63343 7.79285L3.48569 7.88226C3.10108 8.11984 2.71774 8.35614 2.33312 8.62309L2.27454 8.65247L2.36369 8.71123C2.62986 9.06632 2.89603 9.45206 3.1915 9.77778L3.28065 9.92594H3.27938Z" fill="#3DAE2B"></path><path d="M3.36735 10.9631L3.33806 10.7855L3.24891 10.9337C2.98274 11.3194 2.71656 11.7333 2.48095 12.1484L2.45166 12.2072L2.51024 12.2365C2.92415 12.4741 3.36735 12.681 3.81055 12.888L3.95828 12.9467L3.8997 12.7986C3.63353 12.2353 3.4565 11.5838 3.36735 10.9618V10.9631Z" fill="#3DAE2B"></path><path d="M1.41747 8.56306L1.47606 8.53369L1.44677 8.47493C1.12201 7.97167 0.855833 7.43776 0.589659 6.90513L0.531075 6.75696L0.472491 6.9345C0.177025 7.85288 0.0292919 8.77126 0 9.74968V9.92722L0.118441 9.80843C0.503057 9.33455 0.946256 8.91943 1.41875 8.56434L1.41747 8.56306Z" fill="#3DAE2B"></path></g></svg></div><div class="as-title-text">Anaconda Assistant<div class="as-subtitle-text">AI-powered coding, insights and debugging in your notebooks.</div></div></div><div class="as-login-body">To enable the following extensions, create an account or sign in.<ul><li><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none"><path d="M10.125 3.25C10.0312 3.1875 10 3.09375 10 3C10 2.90625 10.0312 2.84375 10.125 2.78125L12 2L12.75 0.15625C12.8125 0.0625 12.9062 0 13 0C13.0938 0 13.1562 0.0625 13.2188 0.15625L14 2L15.8438 2.78125C15.9375 2.84375 16 2.90625 16 3C16 3.09375 15.9375 3.1875 15.8438 3.25L14 4L13.2188 5.875C13.1562 5.96875 13.0938 6 13 6C12.9062 6 12.8125 5.96875 12.75 5.875L12 4L10.125 3.25ZM15.8438 12.7812C15.9375 12.8438 16 12.9062 16 13C16 13.0938 15.9375 13.1875 15.8438 13.25L14 14L13.2188 15.875C13.1562 15.9688 13.0938 16 13 16C12.9062 16 12.8125 15.9688 12.75 15.875L12 14L10.125 13.25C10.0312 13.1875 10 13.0938 10 13C10 12.9062 10.0312 12.8438 10.125 12.7812L12 12L12.75 10.1562C12.8125 10.0625 12.9062 10 13 10C13.0938 10 13.1562 10.0625 13.2188 10.1562L14 12L15.8438 12.7812ZM12 8C12 8.1875 11.875 8.375 11.7188 8.4375L8.1875 10.2188L6.4375 13.75C6.34375 13.9062 6.15625 14 6 14C5.78125 14 5.625 13.9062 5.53125 13.75L3.78125 10.2188L0.25 8.4375C0.09375 8.375 0 8.1875 0 8C0 7.8125 0.09375 7.625 0.25 7.5625L3.78125 5.78125L5.53125 2.28125C5.71875 1.9375 6.25 1.9375 6.4375 2.28125L8.1875 5.78125L11.7188 7.5625C11.875 7.625 12 7.8125 12 8Z" fill="currentColor"></path></svg> Anaconda Assistant<div class="as-version-text">4.0.15</div></li><li class="coming-soon">Coming soon!</li><li class="disabled"><svg xmlns="http://www.w3.org/2000/svg" width="30" height="31" viewBox="0 0 30 31"><path d="M30 4.8125C30 4.82031 30 4.82812 30 4.83594C29.9609 7.48438 23.2578 9.625 15 9.625C6.74219 9.625 0.0390625 7.48438 0 4.83594C0 4.82812 0 4.82031 0 4.8125C0 2.15625 6.71875 0 15 0C23.2812 0 30 2.15625 30 4.8125ZM30 8.4375C30 8.4375 30 8.45312 30 8.46094C29.9609 11.1094 23.2578 13.25 15 13.25C6.74219 13.25 0.0390625 11.1094 0 8.46094V8.4375L0.078125 15.2187C0.078125 15.2187 0.078125 15.25 0.078125 15.2578C0.117188 17.8906 6.78125 20.0234 15 20.0234C23.2188 20.0234 29.8828 17.8906 29.9219 15.2578C29.9219 15.25 29.9219 15.2187 29.9219 15.2187L30 8.4375ZM30 18.4375C30 18.4375 30 18.4531 30 18.4609C29.9609 21.1094 23.2578 23.25 15 23.25C6.74219 23.25 0.0390625 21.1094 0 18.4609V18.4375L0.078125 25.2188C0.078125 25.2188 0.078125 25.25 0.078125 25.2578C0.117188 27.8906 6.78125 30.0234 15 30.0234C23.2188 30.0234 29.8828 27.8906 29.9219 25.2578C29.9219 25.25 29.9219 25.2188 29.9219 25.2188L30 18.4375Z" fill="currentColor"></path></svg> Data Catalogs</li><li class="disabled"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 18"><g clip-path="url(#clip0_345_19040)"><path d="M17.9113 0.428824C17.8716 0.1857 17.6841 0.0656821 17.3871 0.0427376C16.5951 -0.018595 15.8057 -0.0247723 15.0168 0.0970103C14.1202 0.23556 13.2544 0.476919 12.4179 0.823735C11.2931 1.28969 10.265 1.91801 9.31549 2.67783C9.26077 2.72151 9.17297 2.76034 9.10899 2.75019C8.68937 2.68313 8.27328 2.5962 7.8541 2.52648C6.92352 2.37161 6.0481 2.53134 5.27328 3.07362C4.92955 3.3141 4.6348 3.62826 4.33211 3.92345C3.58288 4.65415 2.22253 6.00435 2.09678 6.1279L1.17723 7.17364C1.00603 7.37573 1.10266 7.66651 1.40756 7.72078L2.94661 7.89507C3.66363 7.96125 4.38109 8.02215 5.09855 8.08745C5.14576 8.09186 5.19165 8.11348 5.24062 8.12804C5.09281 8.51987 4.95205 8.89183 4.81218 9.26468C4.73187 9.47868 4.7857 9.65783 4.9472 9.81667C5.25033 10.1145 5.54817 10.4172 5.8491 10.7172C6.6398 11.5071 7.43006 12.2969 8.22253 13.0849C8.3527 13.2147 8.51155 13.2584 8.69025 13.197C8.79526 13.1613 8.89984 13.1229 9.00397 13.0836C9.29475 12.9733 9.58509 12.8621 9.89043 12.7456C9.90057 12.8184 9.90984 12.8687 9.91469 12.9195C10.025 14.0861 10.1353 15.2532 10.2447 16.4199C10.2633 16.6166 10.3401 16.7724 10.5333 16.8492C10.7345 16.929 10.9084 16.8726 11.0544 16.727L14.8628 12.8502C14.9647 12.683 15.0741 12.5308 15.1558 12.3648C15.4695 11.7272 15.6041 11.0526 15.5048 10.3453C15.4386 9.87271 15.338 9.40544 15.2656 8.93375C15.2528 8.84991 15.2806 8.74049 15.3292 8.67033C15.6045 8.27277 15.909 7.89507 16.1715 7.49001C16.8409 6.45795 17.3421 5.34558 17.658 4.15599C17.8667 3.37014 17.9974 2.56929 17.9965 1.75211L18.0004 1.34925C17.9726 1.04215 17.9612 0.731957 17.9118 0.428383L17.9113 0.428824ZM16.2567 5.3769C15.5326 7.04171 14.4617 8.44883 13.1 9.64327C12.484 10.1833 11.82 10.6542 11.1197 11.076C11.0527 11.1161 10.9834 11.1523 10.8669 11.2167C10.5761 10.9114 10.2853 10.595 9.98265 10.2901C8.97044 9.27174 7.95646 8.25512 6.93631 7.24468C6.82159 7.13084 6.81056 7.05097 6.89087 6.91595C7.77511 5.43382 8.88484 4.15378 10.2655 3.11201C11.0632 2.5106 11.9268 2.02876 12.8445 1.63783C13.0329 1.55752 13.1499 1.57385 13.3003 1.72696C14.2199 2.66283 15.1496 3.58855 16.0819 4.51163C16.4649 4.8911 16.4707 4.8836 16.2562 5.37646L16.2567 5.3769Z" fill="currentColor"></path><path d="M11.9237 4.4438C11.016 4.44159 10.2959 5.16214 10.2959 6.07286C10.2959 6.99594 10.9962 7.70104 11.9135 7.70193C12.8269 7.70325 13.5452 6.99329 13.5523 6.08213C13.5589 5.18773 12.822 4.44601 11.9237 4.44336V4.4438Z" fill="currentColor"></path><path d="M0.733782 17.2304C0.573611 17.2304 0.413882 17.1655 0.297394 17.038C0.0776562 16.7971 0.0944234 16.4234 0.335782 16.2036L4.91013 12.0299C5.15105 11.8102 5.52478 11.8274 5.74452 12.0683C5.96426 12.3092 5.94749 12.683 5.70613 12.9027L1.13178 17.0759C1.01838 17.1792 0.875862 17.2304 0.733782 17.2304Z" fill="currentColor"></path><path d="M4.13178 17.3977C3.97161 17.3977 3.81188 17.3328 3.6954 17.2053C3.47566 16.9644 3.49243 16.5906 3.73379 16.3709L5.55699 14.7074C5.79791 14.4877 6.17164 14.5049 6.39138 14.7458C6.61112 14.9867 6.59435 15.3604 6.35299 15.5802L4.52978 17.2437C4.41639 17.3469 4.27386 17.3981 4.13178 17.3981V17.3977Z" fill="currentColor"></path><path d="M0.590823 14.3828C0.430652 14.3828 0.270923 14.3179 0.154435 14.1904C-0.0653034 13.9495 -0.0485362 13.5757 0.192823 13.356L2.01603 11.6925C2.25695 11.4728 2.63024 11.49 2.85042 11.7309C3.07016 11.9718 3.05339 12.3455 2.81203 12.5653L0.988822 14.2288C0.875423 14.332 0.732902 14.3832 0.590823 14.3832V14.3828Z" fill="currentColor"></path></g><defs><clippath id="clip0_345_19040"><rect width="18" height="17.3977" fill="white"></rect></clippath></defs></svg> Panel Deployments</li><li class="disabled"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><g clip-path="url(#clip0_289_5006)"><path d="M18 16.08C17.24 16.08 16.56 16.38 16.04 16.85L8.91 12.7C8.96 12.47 9 12.24 9 12C9 11.76 8.96 11.53 8.91 11.3L15.96 7.19C16.5 7.69 17.21 8 18 8C19.66 8 21 6.66 21 5C21 3.34 19.66 2 18 2C16.34 2 15 3.34 15 5C15 5.24 15.04 5.47 15.09 5.7L8.04 9.81C7.5 9.31 6.79 9 6 9C4.34 9 3 10.34 3 12C3 13.66 4.34 15 6 15C6.79 15 7.5 14.69 8.04 14.19L15.16 18.35C15.11 18.56 15.08 18.78 15.08 19C15.08 20.61 16.39 21.92 18 21.92C19.61 21.92 20.92 20.61 20.92 19C20.92 17.39 19.61 16.08 18 16.08Z" fill="currentColor"></path></g><defs><clippath id="clip0_289_5006"><rect width="24" height="24" fill="white"></rect></clippath></defs></svg> Sharing</li></ul><button class="as-create-account-button">Create Account</button><div class="as-login-sign-in-text">Already have an account? <a>Sign In</a></div></div><div class="as-login-footer"><div class="as-login-help"><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20"><path fill-rule="evenodd" clip-rule="evenodd" d="M18.75 10C18.75 14.832 14.832 18.75 10 18.75C5.16797 18.75 1.25 14.832 1.25 10C1.25 5.16797 5.16797 1.25 10 1.25C14.832 1.25 18.75 5.16797 18.75 10ZM17.5 10C17.5 5.85938 14.1406 2.5 10 2.5C5.85938 2.5 2.5 5.85938 2.5 10C2.5 14.1406 5.85938 17.5 10 17.5C14.1406 17.5 17.5 14.1406 17.5 10ZM13.75 7.5C13.75 9.26562 12.8359 10.0742 12.1641 10.6641C11.5742 11.1836 11.25 11.4727 11.25 12.5H8.75C8.75 10.3438 9.80859 9.40625 10.5117 8.78906C11.0508 8.31641 11.25 8.13672 11.25 7.5C11.25 6.8125 10.6875 6.25 10 6.25C9.3125 6.25 8.75 6.8125 8.75 7.5H6.25C6.25 5.42969 7.92969 3.75 10 3.75C12.0703 3.75 13.75 5.42969 13.75 7.5ZM11.25 16.25V13.75H8.75V16.25H11.25Z" fill="currentColor"></path></svg></div><div class="as-login-footer-text">For more information, read our <a href="https://docs.anaconda.com/free/anaconda-notebooks/anaconda-assistant/" target="_blank">Anaconda Assistant documentation</a>.</div></div></div></div></div></div></div></div></div><div class="lm-SplitPanel-handle lm-mod-hidden" style="position: absolute; contain: style;"></div></div></div><div class="lm-Widget jp-Toolbar-responsive-popup lm-mod-hidden"></div><div class="lm-Widget jp-Completer lm-mod-hidden"></div><div class="lm-Widget lm-Panel jp-ModalCommandPalette lm-mod-hidden" id="modal-command-palette" tabindex="0"><div class="lm-Widget lm-CommandPalette" id="command-palette" role="region" aria-label="Command Palette Section" style=""><div class="lm-CommandPalette-search"><div class="lm-CommandPalette-wrapper"><input class="lm-CommandPalette-input" spellcheck="false" placeholder="SEARCH"><div class="jp-SearchIconGroup"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 18 18" width="16" data-icon="ui-components:search" data-icon-id="62fb2347-7fdb-41a0-b756-53d695ddb513"><g class="jp-icon3" fill="#616161"><path d="M12.1,10.9h-0.7l-0.2-0.2c0.8-0.9,1.3-2.2,1.3-3.5c0-3-2.4-5.4-5.4-5.4S1.8,4.2,1.8,7.1s2.4,5.4,5.4,5.4 c1.3,0,2.5-0.5,3.5-1.3l0.2,0.2v0.7l4.1,4.1l1.2-1.2L12.1,10.9z M7.1,10.9c-2.1,0-3.7-1.7-3.7-3.7s1.7-3.7,3.7-3.7s3.7,1.7,3.7,3.7 S9.2,10.9,7.1,10.9z"></path></g></svg></div><button class="lm-close-icon" style="display: none;"></button></div></div><ul class="lm-CommandPalette-content" role="menu"><li class="lm-CommandPalette-header jp-icon-hoverShow">Assistant<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="assistant:toggle"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Open Anaconda Assistant</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Console<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="console:change-kernel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change Kernel…</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="console:clear"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Clear Console Cells</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="console:close-and-shutdown"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Close and Shut Down…</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="console:linebreak"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Insert Line Break</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="console:interrupt-kernel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Interrupt Kernel</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="console:create"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">New Console</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="console:restart-kernel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Restart Kernel…</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="console:run-forced"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Run Cell (forced)</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item lm-mod-disabled" role="menuitem" data-command="console:run-unforced"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Run Cell (unforced)</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" aria-checked="false" class="lm-CommandPalette-item lm-mod-disabled" data-command="console:toggle-show-all-kernel-activity"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Show All Kernel Activity</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Display Languages<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitemcheckbox" class="lm-CommandPalette-item lm-mod-disabled lm-mod-toggled" data-command="jupyterlab-translation:en" aria-checked="true"><div class="fvaq30 lm-CommandPalette-itemIcon"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:check" data-icon-id="c88e3467-a2a8-4e61-b870-0011e0d7f563"><g class="jp-icon3 jp-icon-selectable" fill="#616161"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></g></svg></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">English</div><div class="lm-CommandPalette-itemCaption">English</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">File Operations<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitemcheckbox" class="lm-CommandPalette-item lm-mod-toggled" data-command="docmanager:toggle-autosave" aria-checked="true"><div class="fvaq30 lm-CommandPalette-itemIcon"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:check" data-icon-id="c88e3467-a2a8-4e61-b870-0011e0d7f563"><g class="jp-icon3 jp-icon-selectable" fill="#616161"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></g></svg></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Autosave Documents</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="docmanager:download"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Download</div><div class="lm-CommandPalette-itemCaption">Download the file to your computer</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="docmanager:reload"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Reload Notebook from Disk</div><div class="lm-CommandPalette-itemCaption">Reload contents from disk</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="docmanager:restore-checkpoint"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Revert Notebook to Checkpoint…</div><div class="lm-CommandPalette-itemCaption">Revert contents to previous checkpoint</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="docmanager:save"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save Notebook</div><div class="lm-CommandPalette-itemCaption">Save and create checkpoint</div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+S</div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="docmanager:save-as"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save Notebook As…</div><div class="lm-CommandPalette-itemCaption">Save with new path</div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Shift+S</div></li><li role="menuitemcheckbox" class="lm-CommandPalette-item lm-mod-disabled" data-command="htmlviewer:trust-html" aria-checked="false"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Trust HTML File</div><div class="lm-CommandPalette-itemCaption">Whether the HTML file is trusted.
    Trusting the file allows scripts to run in it,
    which may result in security risks.
    Only enable for files you trust.</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Help<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="help:about"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">About Jupyter Notebook</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="jupyter-notebook:launch-tree"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Launch Jupyter Notebook File Browser</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="apputils:display-shortcuts"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Show Keyboard Shortcuts</div><div class="lm-CommandPalette-itemCaption">Show relevant keyboard shortcuts for the current active widget</div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Shift+H</div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Image Viewer<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="imageviewer:flip-horizontal"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Flip image horizontally</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">H</div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="imageviewer:flip-vertical"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Flip image vertically</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">V</div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="imageviewer:invert-colors"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Invert Colors</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">I</div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="imageviewer:reset-image"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Reset Image</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">0</div></li><li class="lm-CommandPalette-item lm-mod-disabled" role="menuitem" data-command="imageviewer:rotate-clockwise"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Rotate Clockwise</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">]</div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="imageviewer:rotate-counterclockwise"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Rotate Counterclockwise</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">[</div></li><li class="lm-CommandPalette-item lm-mod-disabled" role="menuitem" data-command="imageviewer:zoom-in"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Zoom In</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">=</div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="imageviewer:zoom-out"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Zoom Out</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">-</div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Kernel Operations<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="kernelmenu:shutdownAll"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Shut Down All Kernels…</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Main Area<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="application:close-other-tabs"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Close All Other Tabs</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="application:close"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Close Tab</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Alt+W</div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="application:close-right-tabs"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Close Tabs to Right</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="documentsearch:end"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">End Search</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Esc</div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="documentsearch:highlightNext"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Find Next</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+G</div></li><li class="lm-CommandPalette-item lm-mod-disabled" role="menuitem" data-command="documentsearch:highlightPrevious"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Find Previous</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Shift+G</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="documentsearch:start"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Find…</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+F</div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="filemenu:logout"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Log Out</div><div class="lm-CommandPalette-itemCaption">Log out of JupyterLab</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="filemenu:shutdown"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Shut Down</div><div class="lm-CommandPalette-itemCaption">Shut down JupyterLab</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Mode<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="application:toggle-zen"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Toggle Zen Mode</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Notebook Cell Operations<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:change-cell-to-code"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change to Code Cell Type</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Y</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:change-cell-to-heading-1"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change to Heading 1</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">1</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:change-cell-to-heading-2"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change to Heading 2</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">2</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:change-cell-to-heading-3"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change to Heading 3</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">3</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:change-cell-to-heading-4"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change to Heading 4</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">4</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:change-cell-to-heading-5"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change to Heading 5</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">5</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:change-cell-to-heading-6"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change to Heading 6</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">6</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:change-cell-to-markdown"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change to Markdown Cell Type</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">M</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:change-cell-to-raw"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change to Raw Cell Type</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">R</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:clear-cell-output"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Clear Cell Output</div><div class="lm-CommandPalette-itemCaption">Clear outputs for the selected cells</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:hide-all-cell-code"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Collapse All Code</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:hide-all-cell-outputs"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Collapse All Outputs</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:hide-cell-code"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Collapse Selected Code</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:hide-cell-outputs"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Collapse Selected Outputs</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:copy-cell"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Copy Cell</div><div class="lm-CommandPalette-itemCaption">Copy this cell</div></div><div class="lm-CommandPalette-itemShortcut">C</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:cut-cell"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Cut Cell</div><div class="lm-CommandPalette-itemCaption">Cut this cell</div></div><div class="lm-CommandPalette-itemShortcut">X</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:delete-cell"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Delete Cell</div><div class="lm-CommandPalette-itemCaption">Delete this cell</div></div><div class="lm-CommandPalette-itemShortcut">D, D</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:disable-output-scrolling"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Disable Scrolling for Outputs</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:enable-output-scrolling"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Enable Scrolling for Outputs</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:show-all-cell-code"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Expand All Code</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:show-all-cell-outputs"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Expand All Outputs</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:show-cell-code"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Expand Selected Code</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:show-cell-outputs"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Expand Selected Outputs</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:extend-marked-cells-above"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Extend Selection Above</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Shift+K</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:extend-marked-cells-below"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Extend Selection Below</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Shift+J</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:extend-marked-cells-bottom"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Extend Selection to Bottom</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Shift+End</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:extend-marked-cells-top"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Extend Selection to Top</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Shift+Home</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:insert-cell-above"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Insert Cell Above</div><div class="lm-CommandPalette-itemCaption">Insert a cell above</div></div><div class="lm-CommandPalette-itemShortcut">A</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:insert-cell-below"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Insert Cell Below</div><div class="lm-CommandPalette-itemCaption">Insert a cell below</div></div><div class="lm-CommandPalette-itemShortcut">B</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:insert-heading-above"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Insert Heading Above Current Heading</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Shift+A</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:insert-heading-below"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Insert Heading Below Current Heading</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Shift+B</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:merge-cell-above"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Merge Cell Above</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Backspace</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:merge-cell-below"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Merge Cell Below</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Shift+M</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:merge-cells"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Merge Selected Cells</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Shift+M</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:move-cell-down"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Move Cell Down</div><div class="lm-CommandPalette-itemCaption">Move this cell down</div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Shift+Down</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:move-cell-up"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Move Cell Up</div><div class="lm-CommandPalette-itemCaption">Move this cell up</div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Shift+Up</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:paste-cell-above"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Paste Cell Above</div><div class="lm-CommandPalette-itemCaption">Paste this cell from the clipboard</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:paste-and-replace-cell"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Paste Cell and Replace</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:paste-cell-below"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Paste Cell Below</div><div class="lm-CommandPalette-itemCaption">Paste this cell from the clipboard</div></div><div class="lm-CommandPalette-itemShortcut">V</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:redo-cell-action"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Redo Cell Operation</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Shift+Z</div></li><li role="menuitemcheckbox" class="lm-CommandPalette-item" data-command="notebook:toggle-render-side-by-side-current" aria-checked="false"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Render Side-by-Side</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Shift+R</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:run-cell-and-select-next"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Run Selected Cell</div><div class="lm-CommandPalette-itemCaption">Run this cell and advance</div></div><div class="lm-CommandPalette-itemShortcut">Shift+Enter</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:run-cell"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Run Selected Cell and Do not Advance</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Enter</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:run-cell-and-insert-below"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Run Selected Cell and Insert Below</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Alt+Enter</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:run-in-console"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Run Selected Text or Current Line in Console</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:move-cursor-up"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Select Cell Above</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">K</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:move-cursor-down"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Select Cell Below</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">J</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:move-cursor-heading-above-or-collapse"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Select Heading Above or Collapse Heading</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Left</div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="notebook:move-cursor-heading-below-or-expand"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Select Heading Below or Expand Heading</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Right</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:set-side-by-side-ratio"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Set side-by-side ratio</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="notebook:split-cell-at-cursor"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Split Cell</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Shift+-</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:undo-cell-action"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Undo Cell Operation</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Z</div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Notebook Operations<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:change-kernel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Change Kernel…</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:clear-all-cell-outputs"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Clear Outputs of All Cells</div><div class="lm-CommandPalette-itemCaption">Clear all outputs of all cells</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:close-and-shutdown"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Close and Shut Down Notebook</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:collapse-all-headings"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Collapse All Headings</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Shift+Left</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:deselect-all"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Deselect All Cells</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:enter-command-mode"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Enter Command Mode</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+M</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:enter-edit-mode"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Enter Edit Mode</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Enter</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:expand-all-headings"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Expand All Headings</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+Shift+Right</div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:interrupt-kernel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Interrupt Kernel</div><div class="lm-CommandPalette-itemCaption">Interrupt the kernel</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:create-console"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">New Console for Notebook</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:create-new"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">New Notebook</div><div class="lm-CommandPalette-itemCaption">Create a new notebook</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:open-with-panel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Open with Panel in New Browser Tab</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:render-with-panel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Preview Notebook with Panel</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:reconnect-to-kernel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Reconnect to Kernel</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:render-all-markdown"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Render All Markdown Cells</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:restart-clear-output"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Restart Kernel and Clear Outputs of All Cells…</div><div class="lm-CommandPalette-itemCaption">Restart the kernel and clear all outputs of all cells</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="debugger:restart-debug"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Restart Kernel and Debug…</div><div class="lm-CommandPalette-itemCaption">Restart Kernel and Debug…</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:restart-run-all"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Restart Kernel and Run All Cells…</div><div class="lm-CommandPalette-itemCaption">Restart the kernel and run all cells</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:restart-and-run-to-selected"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Restart Kernel and Run up to Selected Cell…</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:restart-kernel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Restart Kernel…</div><div class="lm-CommandPalette-itemCaption">Restart the kernel</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:run-all-above"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Run All Above Selected Cell</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:run-all-cells"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Run All Cells</div><div class="lm-CommandPalette-itemCaption">Run all cells</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:run-all-below"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Run Selected Cell and All Below</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: Asciidoc</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: Executable Script</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: HTML</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: LaTeX</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: Markdown</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: PDF</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: Qtpdf</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: Qtpng</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: ReStructured Text</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: Reveal.js Slides</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="notebook:export-to-format"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Save and Export Notebook: Webpdf</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="notebook:select-all"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Select All Cells</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut">Ctrl+A</div></li><li class="lm-CommandPalette-item" role="menuitemcheckbox" aria-checked="false" data-command="notebook:toggle-all-cell-line-numbers"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Show Line Numbers</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item lm-mod-disabled" data-command="notebook:toggle-heading-collapse"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Toggle Collapse Notebook Heading</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitem" data-command="notebook:trust"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Trust Notebook</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Other<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="jupyter-notebook:open-lab"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Open in JupyterLab</div><div class="lm-CommandPalette-itemCaption">JupyterLab</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Terminal<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="terminal:decrease-font"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Decrease Terminal Font Size</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="terminal:increase-font"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Increase Terminal Font Size</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="terminal:create-new"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">New Terminal</div><div class="lm-CommandPalette-itemCaption">Start a new terminal session</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item lm-mod-disabled" role="menuitem" data-command="terminal:refresh"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Refresh Terminal</div><div class="lm-CommandPalette-itemCaption">Refresh the current terminal session</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" class="lm-CommandPalette-item" data-command="terminal:set-theme" aria-checked="false"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Use Terminal Theme: Dark</div><div class="lm-CommandPalette-itemCaption">Set the terminal theme</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item lm-mod-toggled" role="menuitemcheckbox" aria-checked="true" data-command="terminal:set-theme"><div class="fvaq30 lm-CommandPalette-itemIcon"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:check" data-icon-id="c88e3467-a2a8-4e61-b870-0011e0d7f563"><g class="jp-icon3 jp-icon-selectable" fill="#616161"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></g></svg></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Use Terminal Theme: Inherit</div><div class="lm-CommandPalette-itemCaption">Set the terminal theme</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" class="lm-CommandPalette-item" data-command="terminal:set-theme" aria-checked="false"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Use Terminal Theme: Light</div><div class="lm-CommandPalette-itemCaption">Set the terminal theme</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Text Editor<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="fileeditor:change-font-size"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Decrease Font Size</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="fileeditor:change-font-size"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Increase Font Size</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="fileeditor:create-new-markdown-file"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">New Markdown File</div><div class="lm-CommandPalette-itemCaption">Create a new markdown file</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="fileeditor:create-new"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">New Python File</div><div class="lm-CommandPalette-itemCaption">Create a new Python file</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="fileeditor:create-new"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">New Text File</div><div class="lm-CommandPalette-itemCaption">Create a new text file</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitemcheckbox" data-command="fileeditor:change-tabs" aria-checked="false"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Spaces: 1</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" class="lm-CommandPalette-item" data-command="fileeditor:change-tabs" aria-checked="false"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Spaces: 2</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitemcheckbox" aria-checked="false" data-command="fileeditor:change-tabs"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Spaces: 4</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" class="lm-CommandPalette-item" data-command="fileeditor:change-tabs" aria-checked="false"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Spaces: 4</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-item" role="menuitemcheckbox" aria-checked="false" data-command="fileeditor:change-tabs"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Spaces: 8</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">Theme<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="apputils:decr-font-size"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Decrease Code Font Size</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="apputils:decr-font-size"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Decrease Content Font Size</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="apputils:decr-font-size"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Decrease UI Font Size</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="apputils:incr-font-size"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Increase Code Font Size</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="apputils:incr-font-size"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Increase Content Font Size</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="apputils:incr-font-size"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Increase UI Font Size</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" aria-checked="false" class="lm-CommandPalette-item" data-command="apputils:theme-scrollbars"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Theme Scrollbars</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" aria-checked="false" class="lm-CommandPalette-item" data-command="apputils:change-theme"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Use Theme: JupyterLab Dark</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" aria-checked="true" class="lm-CommandPalette-item lm-mod-toggled" data-command="apputils:change-theme"><div class="fvaq30 lm-CommandPalette-itemIcon"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:check" data-icon-id="c88e3467-a2a8-4e61-b870-0011e0d7f563"><g class="jp-icon3 jp-icon-selectable" fill="#616161"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></g></svg></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Use Theme: JupyterLab Light</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li class="lm-CommandPalette-header jp-icon-hoverShow">View<span class="jp-icon-hoverShow-content f1qewoua"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:filter-list" data-icon-id="649d7ffd-f174-409c-9524-83ad9876f20f"><g class="jp-icon3" fill="#616161"><path d="M10 18h4v-2h-4v2zM3 6v2h18V6H3zm3 7h12v-2H6v2z"></path></g></svg></span></li><li role="menuitem" class="lm-CommandPalette-item" data-command="application:open-tree"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">File Browser</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitem" class="lm-CommandPalette-item" data-command="application:open-lab"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Open JupyterLab</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" aria-checked="false" class="lm-CommandPalette-item" data-command="application:toggle-panel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Show Anaconda Assistant</div><div class="lm-CommandPalette-itemCaption">Show Show Anaconda Assistant in the right sidebar</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" aria-checked="true" class="lm-CommandPalette-item lm-mod-toggled" data-command="application:toggle-top"><div class="fvaq30 lm-CommandPalette-itemIcon"><svg xmlns="http://www.w3.org/2000/svg" width="16" viewBox="0 0 24 24" data-icon="ui-components:check" data-icon-id="c88e3467-a2a8-4e61-b870-0011e0d7f563"><g class="jp-icon3 jp-icon-selectable" fill="#616161"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"></path></g></svg></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Show Header</div><div class="lm-CommandPalette-itemCaption"></div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" aria-checked="false" class="lm-CommandPalette-item" data-command="application:toggle-panel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Show Notebook Tools</div><div class="lm-CommandPalette-itemCaption">Show Show Notebook Tools in the right sidebar</div></div><div class="lm-CommandPalette-itemShortcut"></div></li><li role="menuitemcheckbox" aria-checked="false" class="lm-CommandPalette-item" data-command="application:toggle-panel"><div class="fvaq30 lm-CommandPalette-itemIcon"></div><div class="lm-CommandPalette-itemContent"><div class="lm-CommandPalette-itemLabel">Show Table of Contents</div><div class="lm-CommandPalette-itemCaption">Show Show Table of Contents in the left sidebar</div></div><div class="lm-CommandPalette-itemShortcut"></div></li></ul></div></div><link rel="stylesheet" type="text/css" href="file:///C:/Users/CALEB/Downloads/Machine%20learning%20and%20ant%20colony%20optimization/Glaso.ipynb_files/index.css"></body></html>